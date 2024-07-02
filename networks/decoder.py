import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        c1, c2, c3, c4 = x
        tmp = torch.cat((self.conv1(c1), c2), dim=1)
        tmp = self.batchnorm1(tmp)
        tmp = torch.cat((self.conv2(tmp), c3), dim=1)
        tmp = self.batchnorm2(tmp)
        tmp = torch.cat((self.conv3(tmp), c4), dim=1)
        tmp = self.batchnorm3(tmp)
        tmp = self.conv5(tmp)
        tmp = tmp.flatten(1)
        tmp = self.relu(tmp)
        y = self.linear(tmp)
        return y

