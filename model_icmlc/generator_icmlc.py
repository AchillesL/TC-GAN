#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/27 21:42
# @Author : lzg

import torch
import torch.nn as nn
from model_icmlc import basic_block

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        filter_n1 = 64
        filter_list = [filter_n1, filter_n1 * 2, filter_n1 * 4, filter_n1 * 8, filter_n1 * 16]

        self.encoder_block_1 = basic_block.res_block_2(1, filter_list[0])
        self.encoder_block_2 = basic_block.res_block_2(filter_list[0], filter_list[1])
        self.encoder_block_3 = basic_block.res_block_2(filter_list[1], filter_list[2])
        self.encoder_block_4 = basic_block.res_block_2(filter_list[2], filter_list[3])

        self.encoder_block_cswin_1 = basic_block.cswin_transformer_block(filter_list[0], filter_list[0], split_size=8, patch_size=4, reso=128)
        self.encoder_block_cswin_2 = basic_block.cswin_transformer_block(filter_list[1], filter_list[1], split_size=8, patch_size=2, reso=64)
        self.encoder_block_cswin_3 = basic_block.cswin_transformer_block(filter_list[2], filter_list[2], split_size=8, patch_size=2, reso=32)
        self.encoder_block_cswin_4 = basic_block.cswin_transformer_block(filter_list[3], filter_list[3], split_size=8, patch_size=2, reso=16)

        self.downsample_block_1 = basic_block.downsample_block()
        self.downsample_block_2 = basic_block.downsample_block()
        self.downsample_block_3 = basic_block.downsample_block()
        self.downsample_block_4 = basic_block.downsample_block()

        self.upsample_block_4 = basic_block.upsample_block(filter_list[4], filter_list[3])
        self.upsample_block_3 = basic_block.upsample_block(filter_list[3], filter_list[2])
        self.upsample_block_2 = basic_block.upsample_block(filter_list[2], filter_list[1])
        self.upsample_block_1 = basic_block.upsample_block(filter_list[1], filter_list[0])

        self.bottlenet_block = basic_block.conv_basic_block(filter_list[3], filter_list[4])

        self.decoder_block_4 = basic_block.conv_basic_block(filter_list[4], filter_list[3])
        self.decoder_block_3 = basic_block.conv_basic_block(filter_list[3], filter_list[2])
        self.decoder_block_2 = basic_block.conv_basic_block(filter_list[2], filter_list[1])
        self.decoder_block_1 = basic_block.conv_basic_block(filter_list[1], filter_list[0])

        self.tail = nn.Conv2d(filter_list[0], 1, kernel_size=1)

        self.active = nn.Tanh()

    def forward(self, x):
        e_x1 = self.encoder_block_1(x)
        e_x1 = self.encoder_block_cswin_1(e_x1)

        e_x2 = self.downsample_block_1(e_x1)
        e_x2 = self.encoder_block_2(e_x2)
        e_x2 = self.encoder_block_cswin_2(e_x2)

        e_x3 = self.downsample_block_2(e_x2)
        e_x3 = self.encoder_block_3(e_x3)
        e_x3 = self.encoder_block_cswin_3(e_x3)

        e_x4 = self.downsample_block_3(e_x3)
        e_x4 = self.encoder_block_4(e_x4)
        e_x4 = self.encoder_block_cswin_4(e_x4)

        bottle_net = self.downsample_block_4(e_x4)
        bottle_net = self.bottlenet_block(bottle_net)

        d_x4 = self.upsample_block_4(bottle_net)
        d_x4 = torch.cat((e_x4, d_x4), dim=1)
        d_x4 = self.decoder_block_4(d_x4)

        d_x3 = self.upsample_block_3(d_x4)
        d_x3 = torch.cat((e_x3, d_x3), dim=1)
        d_x3 = self.decoder_block_3(d_x3)

        d_x2 = self.upsample_block_2(d_x3)
        d_x2 = torch.cat((e_x2, d_x2), dim=1)
        d_x2 = self.decoder_block_2(d_x2)

        d_x1 = self.upsample_block_1(d_x2)
        d_x1 = torch.cat((e_x1, d_x1), dim=1)
        d_x1 = self.decoder_block_1(d_x1)

        out = self.tail(d_x1)
        out = self.active(out)

        return out



if __name__ == '__main__':
    data = torch.rand(1, 1, 128, 128)
    model = unet()
    model(data)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
