import math
import torch
import torch.nn as nn
from layers.weight_init import trunc_normal_


class DecoderHead(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels[0], in_channels[1], 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels[1]*2, in_channels[2], 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels[2]*2, in_channels[3], 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels[3]*2, 1, 1)
        self.batchnorm1 = norm_layer(in_channels[1]*2)
        self.batchnorm2 = norm_layer(in_channels[2]*2)
        self.batchnorm3 = norm_layer(in_channels[3]*2)
        self.linear = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)
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

    def forward(self, x):
        c1, c2, c3, c4 = x
        tmp = torch.cat((self.conv1(c1), c2), dim=1)
        tmp = self.batchnorm1(tmp)
        tmp = self.relu(tmp)
        tmp = torch.cat((self.conv2(tmp), c3), dim=1)
        tmp = self.batchnorm2(tmp)
        tmp = self.relu(tmp)
        tmp = torch.cat((self.conv3(tmp), c4), dim=1)
        tmp = self.batchnorm3(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv5(tmp)
        tmp = tmp.flatten(1)
        y = self.linear(tmp)
        return y

