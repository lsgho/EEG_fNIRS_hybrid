from torch import nn
import torch
import math
from layers.weight_init import trunc_normal_
from blocks import Block, OverlapPatchEmbed
from fusion import ChannelWeights, CrossBlock
from functools import partial
from torch.nn.functional import interpolate


class HybridTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=None, num_classes=2, embed_dims=None,
                 num_heads=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=None, sr_ratios=None):
        super().__init__()
        if in_chans is None:
            in_chans = [30, 36]
        elif embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        elif num_heads is None:
            num_heads = [1, 2, 4, 8]
        elif mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        elif depths is None:
            depths = [3, 4, 6, 3]
        elif sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans[0],
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans[1],
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                    in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                    in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0] // 2)])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1] // 2)])
        self.extra_norm2 = norm_layer(embed_dims[1])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2] // 2)])
        self.extra_norm3 = norm_layer(embed_dims[2])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3] // 2)])
        self.extra_norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

        self.FRMs = nn.ModuleList([
            ChannelWeights(dim=embed_dims[0], reduction=1),
            ChannelWeights(dim=embed_dims[1], reduction=1),
            ChannelWeights(dim=embed_dims[2], reduction=1),
            ChannelWeights(dim=embed_dims[3], reduction=1)])

        self.FFMs = nn.ModuleList([
            CrossBlock(dim=embed_dims[0], num_heads=num_heads[0], norm_layer=norm_fuse),
            CrossBlock(dim=embed_dims[1], num_heads=num_heads[1], norm_layer=norm_fuse),
            CrossBlock(dim=embed_dims[2], num_heads=num_heads[2], norm_layer=norm_fuse),
            CrossBlock(dim=embed_dims[3], num_heads=num_heads[3], norm_layer=norm_fuse)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x_eeg, x_fnirs):
        B = x_eeg.shape[0]
        outs = []
        outs_fused = []

        # stage 1
        x_eeg, H, W = self.patch_embed1(x_eeg)
        # B H*W/16 C
        x_fnirs, H_, W_ = self.extra_patch_embed1(x_fnirs)
        for i, blk in enumerate(self.block1):
            x_eeg = blk(x_eeg, H, W)
        for i, blk in enumerate(self.extra_block1):
            x_fnirs = blk(x_fnirs, H_, W_)
        x_eeg = self.norm1(x_eeg)
        x_fnirs = self.extra_norm1(x_fnirs)

        x_eeg = x_eeg.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_fnirs = x_fnirs.reshape(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()
        x_eeg, x_fnirs = self.FRMs[0](x_eeg, x_fnirs)
        x_fused = self.FFMs[0](x_eeg, x_fnirs)
        outs.append(x_fused)

        # stage 2
        x_eeg, H, W = self.patch_embed2(x_eeg)
        # B H*W/64 C
        x_fnirs, H_, W_ = self.extra_patch_embed2(x_fnirs)
        for i, blk in enumerate(self.block2):
            x_eeg = blk(x_eeg, H, W)
        for i, blk in enumerate(self.extra_block2):
            x_fnirs = blk(x_fnirs, H_, W_)
        x_eeg = self.norm2(x_eeg)
        x_fnirs = self.extra_norm2(x_fnirs)

        x_eeg = x_eeg.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_fnirs = x_fnirs.reshape(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()
        x_eeg, x_fnirs = self.FRMs[1](x_eeg, x_fnirs)
        x_fused = self.FFMs[1](x_eeg, x_fnirs)
        outs.append(x_fused)

        # stage 3
        x_eeg, H, W = self.patch_embed3(x_eeg)
        # B H*W/256 C
        x_fnirs, H_, W_ = self.extra_patch_embed3(x_fnirs)
        for i, blk in enumerate(self.block3):
            x_eeg = blk(x_eeg, H, W)
        for i, blk in enumerate(self.extra_block3):
            x_fnirs = blk(x_fnirs, H_, W_)
        x_eeg = self.norm3(x_eeg)
        x_fnirs = self.extra_norm3(x_fnirs)

        x_eeg = x_eeg.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_fnirs = x_fnirs.reshape(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()
        x_eeg, x_fnirs = self.FRMs[2](x_eeg, x_fnirs)
        x_fused = self.FFMs[2](x_eeg, x_fnirs)
        outs.append(x_fused)

        # stage 4
        x_eeg, H, W = self.patch_embed4(x_eeg)
        # B H*W/1024 C
        x_fnirs, H_, W_ = self.extra_patch_embed4(x_fnirs)
        for i, blk in enumerate(self.block4):
            x_eeg = blk(x_eeg, H, W)
        for i, blk in enumerate(self.extra_block4):
            x_fnirs = blk(x_fnirs, H_, W_)
        x_eeg = self.norm4(x_eeg)
        x_fnirs = self.extra_norm4(x_fnirs)

        x_eeg = x_eeg.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_fnirs = x_fnirs.reshape(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()
        x_eeg, x_fnirs = self.FRMs[3](x_eeg, x_fnirs)
        x_fused = self.FFMs[3](x_eeg, x_fnirs)
        outs.append(x_fused)
        return outs

    def forward(self, x, y):
        y = interpolate(y, (224, 224), mode='bilinear')
        out = self.forward_features(x, y)
        return out


class mit_b0(HybridTransformer):
    def __init__(self):
        super().__init__(patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                         sr_ratios=[8, 4, 2, 1],drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(HybridTransformer):
    def __init__(self):
        super().__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                         sr_ratios=[8, 4, 2, 1],drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(HybridTransformer):
    def __init__(self):
        super().__init__(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                         sr_ratios=[8, 4, 2, 1],drop_rate=0.0, drop_path_rate=0.1)