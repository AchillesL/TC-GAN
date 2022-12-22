import torch.nn as nn
import torch

"""
------------------------------------------------------------------------------
        Layer (type)    map size      start       jump receptive_field 
==============================================================================
        0             [128, 128]        0.5        1.0             1.0 
        1               [64, 64]        0.5        2.0             5.0 
        2               [32, 32]        0.5        4.0            13.0 
        3               [16, 16]        0.5        8.0            29.0 
        4               [14, 14]        8.5        8.0            61.0 
        5               [12, 12]       16.5        8.0            93.0 
==============================================================================
Total number of parameters : 4.319 M
"""


class Discriminator(nn.Module):

    def __init__(self, n_first_num=2):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(n_first_num, 64, kernel_size=5, stride=2, padding=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.leaky_relu2 = nn.LeakyReLU(0.2, True)
        self.batch_norm2d_2 = torch.nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.leaky_relu3 = nn.LeakyReLU(0.2, True)
        self.batch_norm2d_3 = torch.nn.BatchNorm2d(num_features=256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=1, bias=False)
        self.leaky_relu4 = nn.LeakyReLU(0.2, True)
        self.batch_norm2d_4 = torch.nn.BatchNorm2d(num_features=512)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=5, stride=1, padding=1)

    def forward(self, input):
        """Standard forward."""
        out = self.conv1(input)

        out = self.conv2(out)
        out = self.batch_norm2d_2(out)
        out = self.leaky_relu2(out)

        out = self.conv3(out)
        out = self.batch_norm2d_3(out)
        out = self.leaky_relu3(out)

        out = self.conv4(out)
        out = self.batch_norm2d_4(out)
        out = self.leaky_relu4(out)

        out = self.conv5(out)

        return out

if __name__ == '__main__':
    data = torch.rand(1, 2, 128, 128)
    model = Discriminator()
    model(data)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
