import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class ResidualModule(nn.Module):
    """docstring for ResidualModule."""
    def __init__(self, in_channels, out_channels):
        super(ResidualModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        out_channels_half = out_channels // 2

        # for skip layer
        if in_channels != out_channels:
            self.conv0 = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.bn0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels_half, (1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels_half)
        self.conv2 = nn.Conv2d(out_channels_half, out_channels_half, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels_half)
        self.conv3 = nn.Conv2d(out_channels_half, out_channels, (1, 1))
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        # reshaping
        if self.in_channels != self.out_channels:
            x = self.bn0(self.conv0(x))

        h = h + x
        return h


class HGBlock(nn.Module):
    """docstring for HGBlock."""
    def __init__(self, in_channels, out_channels):
        super(HGBlock, self).__init__()
        self.res1 = ResidualModule(in_channels, 256)
        self.res2 = ResidualModule(256, 256)
        self.res3 = ResidualModule(256, 256)

    def forward(self, x):
        pass


class StackedHG(nn.Module):
    """docstring for StackedHG."""
    def __init__(self, out_channels):
        super(StackedHG, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.res0 = ResidualModule(64, 128)

        self.res1_1 = ResidualModule(128, 128)
        self.res1_2 = ResidualModule(128, 128)
        self.res1_3 = ResidualModule(128, 256)
        self.hg1 = HGBlock(256, 512)

        self.conv1_1 = nn.Conv2d(512, 512, (1, 1))
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512, 256, (1, 1))
        self.bn1_2 = nn.BatchNorm2d(256)

        self.conv1_3 = nn.Conv2d(256, out_channels, (1, 1))
        self.conv1_4 = nn.Conv2d(out_channels, 256+128, (1, 1))

        self.hg2 = HGBlock(256+128, 512)

    def forward(self, x):
        # (256, 256, 3) -> (128, 128, 64)
        h = F.relu(self.bn0(self.conv0(x)))
        # (128, 128, 64) -> (64, 64, 128)
        in_1 = _max_pooling_2d(self.res0(h))

        h = self.res1_1(h)
        h = self.res1_2(h)
        h = self.res1_3(h)
        h = self.hg1(h)

        # l1 and l2
        h = self.bn1_1(self.conv1_1(h))
        h = self.bn1_2(self.conv1_2(h))

        out_1 = self.conv1_3(h)
        h = self.conv1_4(h) +


def _max_pooling_2d(x):
    return F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
