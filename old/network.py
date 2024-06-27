import torch
from torch import nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from torch.nn.functional import cross_entropy, softmax, dropout, cosine_similarity
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from argument import args
import math


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.depthwiseconv2d = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), groups=4, padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.separableconv2d = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 3), padding=(0, 1), groups=8),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = self.depthwiseconv2d(x)
        x = self.separableconv2d(x)
        return x


class NIRSNet(nn.Module):
    def __init__(self):
        super(NIRSNet, self).__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
        )

        self.depthwiseconv2d = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), padding=(0, 1), groups=4),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.separableconv2d = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 3), padding=(0, 1), groups=8),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.conv2d1(x)
        else:
            x = self.conv2d2(x)
        x = self.depthwiseconv2d(x)
        x = self.separableconv2d(x)
        return x


class ClipLossModel(nn.Module):
    def __init__(self):
        super(ClipLossModel, self).__init__()
        self.eeg_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(args.eeg_channel, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Linear(in_features=20*args.eeg_freq, out_features=1024),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.BatchNorm1d(1024)
        )
        self.nirs_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(args.nirs_channel-6, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Linear(in_features=20*args.nirs_freq, out_features=1024),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.BatchNorm1d(1024)
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    def forward(self, eeg_data, nirs_data):
        eeg_data = self.eeg_conv(eeg_data)
        nirs_data = self.nirs_conv(nirs_data)

        eeg_data = eeg_data / eeg_data.norm(p=2, dim=-1, keepdim=True)
        nirs_data = nirs_data / nirs_data.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logit_per_eeg = torch.matmul(eeg_data, nirs_data.t()) * logit_scale
        loss = self.clip_loss(logit_per_eeg)
        return loss


class EEGLossModel(nn.Module):
    def __init__(self):
        super(EEGLossModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(args.eeg_channel, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=20*args.eeg_freq, out_features=10*args.eeg_freq),
            nn.BatchNorm1d(10*args.eeg_freq),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=10*args.eeg_freq, out_features=args.eeg_freq),
            nn.BatchNorm1d(args.eeg_freq),
            nn.ReLU(),
            nn.Linear(in_features=args.eeg_freq, out_features=2),
        )

    def forward(self, data, labels):
        data = self.conv(data)
        data = self.linear(data)
        return cross_entropy(data, labels)


class NIRSLossModel(nn.Module):
    def __init__(self):
        super(NIRSLossModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(args.nirs_channel-6, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)')
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=20*args.nirs_freq, out_features=10*args.nirs_freq),
            nn.BatchNorm1d(10*args.nirs_freq),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=10*args.nirs_freq, out_features=args.nirs_freq),
            nn.BatchNorm1d(args.nirs_freq),
            nn.ReLU(),
            nn.Linear(in_features=args.nirs_freq, out_features=2),
        )

    def forward(self, data, labels):
        data = self.conv(data)
        data = self.linear(data)
        return cross_entropy(data, labels)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim, p):
        super(MultiHeadAttention, self).__init__()
        assert dim % n_head == 0, 'n_head should be divisible by dim'
        self.n_head = n_head
        self.p = p
        self.W_Q = nn.Linear(in_features=dim, out_features=dim)
        self.W_K = nn.Linear(in_features=dim, out_features=dim)
        self.W_V = nn.Linear(in_features=dim, out_features=dim)
        self.layer_norm1 = nn.LayerNorm(dim // n_head)
        self.layer_norm2 = nn.LayerNorm(dim // n_head)

    def forward(self, q, k, v):
        Q = rearrange(self.W_Q(q), 'b h (n d) -> b n h d', n=self.n_head)
        K = rearrange(self.W_K(k), 'b h (n d) -> b n h d', n=self.n_head)
        V = rearrange(self.W_V(v), 'b h (n d) -> b n h d', n=self.n_head)

        Q = self.layer_norm1(Q)
        K = self.layer_norm2(K)
        attn = self.scaled_dot_product_attention(Q, K, V, self.p)
        attn = rearrange(attn, 'b n h d -> b h (n d)')
        return attn

    def scaled_dot_product_attention(self, q, k, v, p=0.1):
        d_k = q.shape[-1]
        q = q / math.sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-1, -2))
        attn = softmax(scores, dim=-1)
        if p > 0.0:
            attn = dropout(attn, p=p)
        output = torch.matmul(attn, v)
        return output


class Transformer(nn.Module):
    def __init__(self, n_head, dim, p=0.1):
        super(Transformer, self).__init__()
        self.n_head = n_head
        self.multi_head_attention = MultiHeadAttention(n_head, dim, p=p)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.feed_forward = nn.Linear(dim, dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        origin = x
        x = self.multi_head_attention(x, x, x)
        x = self.layer_norm1(x + origin)
        tmp = x
        x = self.feed_forward(x)
        x = self.layer_norm2(x + tmp)
        return x

class EEGTransformerNet(nn.Module):
    def __init__(self, n_head, p=0.1):
        super(EEGTransformerNet, self).__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5*args.eeg_freq, kernel_size=(5, args.eeg_freq), stride=(5, args.eeg_freq)),
            Rearrange('b c h w -> b (h w) c')
        )
        self.temporal_embed = nn.Parameter(torch.randn(1, 120+1, 5*args.eeg_freq), requires_grad=True)
        self.spatial_embed = nn.Parameter(torch.randn(1, 120+1, 5*args.eeg_freq), requires_grad=True)
        self.transformer = Transformer(n_head, 5*args.eeg_freq, p)
        self.cls_token = nn.Parameter(torch.randn(1, 5*args.eeg_freq), requires_grad=True)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = repeat(self.cls_token, 'n d -> b n d', b=x.shape[0])
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.temporal_embed + self.spatial_embed
        x = self.transformer(x)
        return x[:, 0]


class NIRSTransformerNet(nn.Module):
    def __init__(self, n_head, p=0.1):
        super(NIRSTransformerNet, self).__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5*args.nirs_freq, kernel_size=(5, args.nirs_freq), stride=(5, args.nirs_freq)),
            Rearrange('b c h w -> b (h w) c')
        )
        self.temporal_embed = nn.Parameter(torch.randn(1, 120+1, 5*args.nirs_freq), requires_grad=True)
        self.spatial_embed = nn.Parameter(torch.randn(1, 120+1, 5*args.nirs_freq), requires_grad=True)
        self.transformer = Transformer(n_head, 5*args.nirs_freq, p)
        self.cls_token = nn.Parameter(torch.randn(1, 5*args.nirs_freq), requires_grad=True)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = repeat(self.cls_token, 'n d -> b n d', b=x.shape[0])
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.temporal_embed + self.spatial_embed
        x = self.transformer(x)
        return x[:, 0]


class CrossModalGuiding(nn.Module):
    def __init__(self):
        super(CrossModalGuiding, self).__init__()
        self.eeg_conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1))
        self.nirs_conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.Linear(20*args.eeg_freq, 20*args.nirs_freq),
            nn.Sigmoid()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.Linear(20*args.eeg_freq, 20*args.nirs_freq),
            nn.Sigmoid()
        )

    def forward(self, eeg, nirs):
        eeg = self.eeg_conv(eeg)
        nirs = self.nirs_conv(nirs)
        tmp_nirs = torch.mul(self.block1(eeg), nirs)              # shape!!!
        nirs = tmp_nirs + self.block2(eeg)
        return [eeg, nirs]


class MySequential(nn.Sequential):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, *input):
        for module in self:
            input = module(*input)
        return input


class CrossModalFusionNet(nn.Module):
    def __init__(self):
        super(CrossModalFusionNet, self).__init__()
        self.eeg_pre = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
        self.nirs_pre = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.cross_modal_guiding = MySequential(*[CrossModalGuiding() for _ in range(3)])
        self.eeg_linear = nn.Linear(20*args.eeg_freq, 20*args.nirs_freq)
        self.nirs_linear = nn.Linear(20*args.nirs_freq, 20*args.eeg_freq)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.Sigmoid()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 3), padding=(0, 1)),
            nn.Sigmoid()
        )
        self.eeg_out = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(args.eeg_channel, 1)),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(in_features=20*args.eeg_freq, out_features=5*args.eeg_freq),
            nn.Tanh()
        )
        self.nirs_out = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(args.nirs_channel-6, 1)),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(in_features=20*args.nirs_freq, out_features=5*args.nirs_freq),
            nn.Tanh()
        )

    def forward(self, eeg_data, nirs_data):
        eeg_data = self.eeg_pre(eeg_data)
        nirs_data = self.nirs_pre(nirs_data)
        eeg_data, nirs_data = self.cross_modal_guiding(eeg_data, nirs_data)

        tmp_nirs = torch.cat((nirs_data, self.eeg_linear(eeg_data)), dim=1)
        tmp_eeg = torch.cat((eeg_data, self.nirs_linear(nirs_data)), dim=1)
        tmp_nirs = self.block1(tmp_nirs)
        tmp_eeg = self.block2(tmp_eeg)

        eeg_data = torch.mul(eeg_data, tmp_eeg)
        nirs_data = torch.mul(nirs_data, tmp_nirs)
        eeg_data = self.eeg_out(eeg_data)
        nirs_data = self.nirs_out(nirs_data)

        result = torch.cat((eeg_data, nirs_data), dim=-1)
        return result


# 搜索融合方法，直接concat感觉不太好
class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.eeg_net = EEGNet()
        self.nirs_net = NIRSNet()
        self.clip_loss = ClipLossModel()
        self.eeg_loss = EEGLossModel()
        self.nirs_loss = NIRSLossModel()

        self.eeg_trans = EEGTransformerNet(n_head=5)
        self.nirs_trans = NIRSTransformerNet(n_head=5)
        self.cross_modal_fusion = CrossModalFusionNet()

        dim = 256
        self.eeg_pre = nn.Linear(5*args.eeg_freq, dim)
        self.nirs_pre = nn.Linear(5*args.nirs_freq, dim)
        self.fusion_pre = nn.Linear(5*args.eeg_freq+5*args.nirs_freq, dim)

        self.eeg_linear1 = nn.Linear(dim, 1)
        self.nirs_linear1 = nn.Linear(dim, 1)
        self.fusion_linear1 = nn.Linear(dim, 1)

        self.eeg_nlinear = nn.Linear(2*dim, dim)
        self.eeg_flinear = nn.Linear(2*dim, dim)
        self.nirs_flinear = nn.Linear(2*dim, dim)

        self.out = nn.Linear(2*dim, 2)

    def forward(self, eeg_data, nirs_data, labels):
        eeg_data = self.eeg_net(eeg_data)
        nirs_data = self.nirs_net(nirs_data)

        loss = self.clip_loss(eeg_data, nirs_data) + self.eeg_loss(eeg_data, labels) + self.nirs_loss(nirs_data, labels)

        eeg_nirs = self.cross_modal_fusion(eeg_data, nirs_data)
        eeg_single = self.eeg_trans(eeg_data)
        nirs_single = self.nirs_trans(nirs_data)

        eeg_nirs = self.fusion_pre(eeg_nirs)
        eeg_single = self.eeg_pre(eeg_single)
        nirs_single = self.nirs_pre(nirs_single)

        result = self.get_multimodal(eeg_single, nirs_single, eeg_nirs)
        loss += cross_entropy(result, labels)
        return loss, result


    def get_multimodal(self, eeg_data, nirs_data, fusion_data):
        alpha1, alpha2, alpha3 = self.eeg_linear1(eeg_data), self.nirs_linear1(nirs_data), self.fusion_linear1(fusion_data)
        u = (alpha1 * eeg_data + alpha2 * nirs_data + alpha3 * fusion_data) / 3

        n_eeg = softmax(eeg_data, dim=-1)
        n_nirs = softmax(nirs_data, dim=-1)
        n_fusion = softmax(fusion_data, dim=-1)

        s_eeg_nirs = torch.matmul(n_eeg, n_nirs.t())
        s_eeg_fusion = torch.matmul(n_eeg, n_fusion.t())
        s_nirs_fusion = torch.matmul(n_nirs, n_fusion.t())
        s_eeg_nirs = torch.diag(s_eeg_nirs).unsqueeze(-1)
        s_eeg_fusion = torch.diag(s_eeg_fusion).unsqueeze(-1)
        s_nirs_fusion = torch.diag(s_nirs_fusion).unsqueeze(-1)

        alpha12_ = (alpha1 + alpha2) / (s_eeg_nirs + 0.5)
        alpha13_ = (alpha1 + alpha3) / (s_eeg_fusion + 0.5)
        alpha23_ = (alpha2 + alpha3) / (s_nirs_fusion + 0.5)

        alpha12 = torch.exp(alpha12_) / (torch.exp(alpha13_) + torch.exp(alpha23_))
        alpha13 = torch.exp(alpha13_) / (torch.exp(alpha23_) + torch.exp(alpha12_))
        alpha23 = torch.exp(alpha23_) / (torch.exp(alpha12_) + torch.exp(alpha13_))
        b = alpha12 * self.eeg_nlinear(torch.cat((eeg_data, nirs_data), dim=-1)) + \
            alpha13 * self.eeg_flinear(torch.cat((eeg_data, fusion_data), dim=-1)) + \
            alpha23 * self.nirs_flinear(torch.cat((nirs_data, fusion_data), dim=-1))

        m = torch.cat((u, b), dim=-1)
        result = self.out(m)
        return result

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
