#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/12/02 20:08
# @Author : lzg

import torch.nn as nn
from timm.models.layers import to_2tuple
from einops.layers.torch import Rearrange
from model_icmlc import cswin

class res_block_2(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(res_block_2, self).__init__()
        self.need_expand_channel = input_channel != output_channel
        self.conv1x1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=True)
        self.conv3x3_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3x3_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.BN_identity = nn.BatchNorm2d(output_channel)
        self.BN1 = nn.BatchNorm2d(input_channel)
        self.BN2 = nn.BatchNorm2d(output_channel)

        self.active = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.need_expand_channel:
            identity = self.conv1x1(x)
            identity = self.BN_identity(identity)

        x = self.BN1(x)
        x = self.active(x)
        x = self.conv3x3_1(x)

        x = self.BN2(x)
        x = self.active(x)
        x = self.conv3x3_2(x)

        out = identity + x
        return out

class downsample_block(nn.Module):
    def __init__(self):
        super(downsample_block, self).__init__()

        self.module_list = nn.ModuleList()

        self.module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

class upsample_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(upsample_block, self).__init__()

        self.module_list = nn.ModuleList()

        self.module_list.append(nn.Upsample(scale_factor=2))
        self.module_list.append(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_list.append(nn.BatchNorm2d(output_channel))
        self.module_list.append(nn.LeakyReLU(0.2, True))

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

class conv_basic_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(conv_basic_block, self).__init__()
        self.conv3x3_1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.active = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.conv3x3_1(x)
        x = self.bn1(x)
        x = self.active(x)

        x = self.conv3x3_2(x)
        x = self.bn2(x)
        x = self.active(x)

        return x

class cswin_transformer_block(nn.Module):
    def __init__(self, dim, num_heads, reso, split_size, patch_size):
        super(cswin_transformer_block, self).__init__()

        self.patch_embed = PatchEmbed(img_size=reso, patch_size=patch_size, in_chans=dim)
        self.patch_unembed = PatchUnEmbed(img_size=reso, patch_size=patch_size, in_chans=dim)

        self.cswin_block = cswin.CSWinBlock(
            dim=self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1] * dim,
            reso=self.patch_embed.patches_resolution[0],
            num_heads=num_heads,
            split_size=split_size,
            drop=0.3,
            drop_path=0.3
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cswin_block(x)
        x = self.patch_unembed(x)

        return x

class PatchEmbed(nn.Module):

    def __init__(self, img_size=128, patch_size=4, in_chans=3, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        if patch_size[0] == 16:
            self.conv = nn.Conv2d(in_chans, in_chans * patch_size[0] * patch_size[1], 16, 16, 0)
        elif patch_size[0] == 8:
            self.conv = nn.Conv2d(in_chans, in_chans * patch_size[0] * patch_size[1], 8, 8, 0)
        elif patch_size[0] == 4:
            self.conv = nn.Conv2d(in_chans, in_chans * patch_size[0] * patch_size[1], 7, 4, 3)
        elif patch_size[0] == 2:
            self.conv = nn.Conv2d(in_chans, in_chans * patch_size[0] * patch_size[1], 3, 2, 1)
        elif patch_size[0] == 1:
            self.conv = nn.Conv2d(in_chans, in_chans * patch_size[0] * patch_size[1], 1, 1, 0)

        self.rearrange = Rearrange('b c h w -> b (h w) c',
                                   h=patches_resolution[0],
                                   w=patches_resolution[1],
                                   c=in_chans * patch_size[0] * patch_size[1])

        if norm_layer is not None:
            self.norm = norm_layer(in_chans)
        else:
            self.norm = nn.LayerNorm(in_chans * patch_size[0] * patch_size[1])

    def forward(self, x):
        x = self.conv(x)
        x = self.rearrange(x)
        x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=128, patch_size=4, in_chans=3):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (p1 w) (p2 h)',
                                   h=patches_resolution[0],
                                   w=patches_resolution[1],
                                   p1=patch_size[0],
                                   p2=patch_size[1],
                                   c=in_chans)

    def forward(self, x):
        x = self.rearrange(x)
        return x