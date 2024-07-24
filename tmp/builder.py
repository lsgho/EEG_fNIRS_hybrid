import torch
import torch.nn as nn
from config import config
from vit import VisionEncoder
from patch_embed import PatchEmbed
from fusion import VisionDecoder, FeatureRectifyModule, CrossBlock
from typing import Tuple, List, Union, Callable

class HybridNet(nn.Module):
    def __init__(
            self,
            in_chans: int = 1,
            pre_norm: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            embed_layer: Callable = PatchEmbed
    ):
        super().__init__()
        eeg_img_size = (config.eeg_channels, config.eeg_fs * 10)
        nirs_img_size = (config.nirs_channels, config.nirs_fs * 10)
        eeg_patch_size = (5, config.eeg_fs)
        nirs_patch_size = (6, config.nirs_fs)
        eeg_embed_dim = 5 * config.eeg_fs
        nirs_embed_dim = 6 * config.nirs_fs
        self.eeg_en1 = VisionEncoder(img_size=(config.eeg_channels, config.eeg_fs * 10), patch_size=(5, config.eeg_fs), embed_dim=5*config.eeg_fs)
        self.nirs_en1 = VisionEncoder(img_size=(config.nirs_channels, config.nirs_fs * 10), patch_size=(6, config.nirs_fs), embed_dim=6*config.nirs_fs)
        self.eeg_en2 = VisionEncoder(img_size=(config.eeg_channels, config.eeg_fs * 10), patch_size=(5, config.eeg_fs), embed_dim=5*config.eeg_fs)
        self.nirs_en2 = VisionEncoder(img_size=(config.nirs_channels, config.nirs_fs * 10), patch_size=(6, config.nirs_fs), embed_dim=6*config.nirs_fs)
        self.eeg_en3 = VisionEncoder(img_size=(config.eeg_channels, config.eeg_fs * 10), patch_size=(5, config.eeg_fs), embed_dim=5*config.eeg_fs)
        self.nirs_en3 = VisionEncoder(img_size=(config.nirs_channels, config.nirs_fs * 10), patch_size=(6, config.nirs_fs), embed_dim=6*config.nirs_fs)
        self.eeg_en4 = VisionEncoder(img_size=(config.eeg_channels, config.eeg_fs * 10), patch_size=(5, config.eeg_fs), embed_dim=5*config.eeg_fs)
        self.nirs_en4 = VisionEncoder(img_size=(config.nirs_channels, config.nirs_fs * 10), patch_size=(6, config.nirs_fs), embed_dim=6*config.nirs_fs)
        self.decoder = VisionDecoder(num_classes=2, dim=5*config.eeg_fs)
        # self.rectify1 = FeatureRectifyModule()
        self.cross_block1 = CrossBlock(dim=[5*config.eeg_fs, 6*config.nirs_fs])
#         self.rectify2 = FeatureRectifyModule()
        self.cross_block2 = CrossBlock(dim=[5*config.eeg_fs, 6*config.nirs_fs])
#         self.rectify3 = FeatureRectifyModule()
        self.cross_block3 = CrossBlock(dim=[5*config.eeg_fs, 6*config.nirs_fs])
#         self.rectify4 = FeatureRectifyModule()
        self.cross_block4 = CrossBlock(dim=[5*config.eeg_fs, 6*config.nirs_fs])


        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed1 = embed_layer(
            img_size=eeg_img_size,
            patch_size=eeg_patch_size,
            in_chans=in_chans,
            embed_dim=eeg_embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )

        self.patch_embed2 = embed_layer(
            img_size=nirs_img_size,
            patch_size=nirs_patch_size,
            in_chans=in_chans,
            embed_dim=nirs_embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_embed1(x)
        y = self.patch_embed2(y)
        x = self.eeg_en1(x)
        y = self.nirs_en1(y)
        # x, y = self.rectify1(x, y)
        c1 = self.cross_block1(x, y)

        x = self.eeg_en2(x[:, 1:, :])
        y = self.nirs_en2(y[:, 1:, :])
        # x, y = self.rectify2(x, y)
        c2 = self.cross_block2(x, y)

        x = self.eeg_en3(x[:, 1:, :])
        y = self.nirs_en3(y[:, 1:, :])
        # x, y = self.rectify3(x, y)
        c3 = self.cross_block3(x, y)

        x = self.eeg_en4(x[:, 1:, :])
        y = self.nirs_en4(y[:, 1:, :])
        # x, y = self.rectify4(x, y)
        c4 = self.cross_block4(x, y)

        scale_image = [c1[:, 0, :].squeeze(1), c2[:, 0, :].squeeze(1), c3[:, 0, :].squeeze(1), c4[:, 0, :].squeeze(1)]
        if label is not None:
            loss = self.decoder(scale_image, label)
            return loss
        else:
            result = self.decoder(scale_image)
            return result

