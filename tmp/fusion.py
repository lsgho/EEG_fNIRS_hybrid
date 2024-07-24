import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mlp import Mlp
from drop import DropPath
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List, Literal
from layers.weight_init import trunc_normal_, lecun_normal_
from vit import LayerScale


def init_weights(module: nn.Module) -> None:
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(
            self,
            dim: int,
            reduction: int = 1
    ) -> None:
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2),
                    nn.Sigmoid())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(
            self,
            dim: int,
            reduction: int = 1
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
                    nn.Sigmoid())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(
            self,
            dim: int = 768,
            reduction: int = 1,
            lambda_c: float = .5,
            lambda_s: float = .5
    ) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.apply(init_weights)

    def init_weights(self, module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
        if isinstance(module, nn.Linear):
            if name.startswith('head'):
                nn.init.zeros_(module.weight)
                nn.init.constant_(module.bias, head_bias)
            else:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return [out_x1, out_x2]


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: List,
            num_heads: int = 5,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim[0] % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim[0] // num_heads
        self.scale = self.head_dim ** -0.5

        self.q1 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.q2 = nn.Linear(dim[1], dim[0], bias=qkv_bias)
        self.kv1 = nn.Linear(dim[1], dim[0] * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim[0], dim[0] * 2, bias=qkv_bias)
        self.q_norm1 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.q_norm2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm1 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim[0], dim[0])
        self.proj2 = nn.Linear(dim[0], dim[0])
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, N, C = x1.shape
        # x1->q
        q = self.q1(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv1(x2).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv
        q, k = self.q_norm1(q), self.k_norm1(k)

        q = q * self.scale
        atten = q @ k.transpose(-2, -1)
        atten = atten.softmax(dim=-1)
        attn = self.attn_drop(atten)
        y1 = attn @ v
        y1 = y1.transpose(1, 2).reshape(B, N, C)
        y1 = self.proj1(y1)
        y1 = self.proj_drop(y1)

        # x2->q
        q = self.q2(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv2(x1).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv
        q, k = self.q_norm2(q), self.k_norm2(k)

        q = q * self.scale
        atten = q @ k.transpose(-2, -1)
        atten = atten.softmax(dim=-1)
        attn = self.attn_drop(atten)
        y2 = attn @ v
        y2 = y2.transpose(1, 2).reshape(B, N, C)
        y2 = self.proj2(y2)
        y2 = self.proj_drop(y2)

        y = torch.cat((y1, y2), dim=-1)
        return y


class CrossBlock(nn.Module):
    def __init__(
            self,
            dim: List,
            num_heads: int = 5,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim[0])
        self.norm1_ = norm_layer(dim[1])
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim[0], init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(2*dim[0])
        self.mlp = mlp_layer(
            in_features=2*dim[0],
            hidden_features=int(2*dim[0] * mlp_ratio),
            out_features=dim[0],
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim[0], init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tmp = self.ls1(self.attn(self.norm1(x), self.norm1_(y)))
        x_ = self.drop_path1(tmp)
        x_ = self.drop_path2(self.ls2(self.mlp(self.norm2(x_))))
        return x_


class VisionDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            dim: int = 768,
            drop_rate: float = 0.,
            act_layer = nn.ReLU
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(drop_rate)
        self.linear2 = nn.Linear(dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = act_layer()
        self.apply(init_weights)

    def forward(self, x: List, label: torch.Tensor = None) -> torch.Tensor:
        c1, c2, c3, c4 = x
        x = torch.cat((c1, c2, c3, c4), dim=-1)
        x = self.linear1(x)
        x = self.acc_fn(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.acc_fn(x)
        x = self.drop(x)
        if label is None:
            return x
        else:
            loss = self.loss_fn(x, label)
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()