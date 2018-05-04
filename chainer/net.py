import chainer
from chainer import functions as F
from chainer import links as L


class ResidualModule(chainer.Chain):
    """docstring for ResidualModule."""
    def __init__(self, in_channels, out_channels):
        super(ResidualModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        out_channels_half = out_channels // 2

        with self.init_scope():
            # for skip layer
            if in_channels != out_channels:
                self.conv0 = L.Convolution2D(in_channels, out_channels, (1, 1))
                self.bn0 = L.BatchNormalization(out_channels)

            self.conv1 = L.Convolution2D(in_channels, out_channels_half, (1, 1))
            self.bn1 = L.BatchNormalization(out_channels_half)
            self.conv2 = L.Convolution2D(out_channels_half, out_channels_half, (3, 3), pad=1)
            self.bn2 = L.BatchNormalization(out_channels_half)
            self.conv3 = L.Convolution2D(out_channels_half, out_channels, (1, 1))
            self.bn3 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        # residual
        if self.in_channels != self.out_channels:
            x = self.bn0(self.conv0(x))

        h = h + x
        return h


class HGBlock(chainer.Chain):
    """docstring for HGBlock."""
    def __init__(self, in_channels, out_channels, layer_num=4):
        super(HGBlock, self).__init__()

        with self.init_scope():
            # residual/upper branch
            self.res1a_1 = ResidualModule(in_channels, 256)
            self.res1a_2 = ResidualModule(256, 256)
            self.res1a_3 = ResidualModule(256, out_channels)

            # lower1 branch
            self.res1b_1 = ResidualModule(in_channels, 256)
            self.res1b_2 = ResidualModule(256, 256)
            self.res1b_3 = ResidualModule(256, 256)

            # lower2
            if layer_num != 1:
                self.block = HGBlock(256, out_channels, layer_num-1)

            else:
                self.block = ResidualModule(256, out_channels)

            self.res1b_4 = ResidualModule(out_channels, out_channels)

    def __call__(self, x):

        N, C, H, W = x.shape

        # 64 x 64
        h_res = self.res1a_3(self.res1a_2(self.res1a_1(x)))
        # 32 x 32
        h = self.res1b_3(self.res1b_2(self.res1b_1(_max_pooling_2d(x))))
        h = self.block(h)

        # 32 x 32 -> 64 x 64
        # FIXME: in original paper NN, but here we use bilinear.
        h = F.resize_images(self.res1b_4(h), (H, W))
        return h + h_res


class StackedHG(chainer.Chain):
    """docstring for StackedHG."""
    def __init__(self, out_channels):
        super(StackedHG, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(3, 64, ksize=(7, 7), stride=2, pad=3)
            self.bn0 = L.BatchNormalization(64)
            self.res0 = ResidualModule(64, 128)

            self.res1_1 = ResidualModule(128, 128)
            self.res1_2 = ResidualModule(128, 128)
            self.res1_3 = ResidualModule(128, 256)
            self.hg1 = HGBlock(256, 512)

            self.conv1_1 = L.Convolution2D(512, 512, (1, 1))
            self.bn1_1 = L.BatchNormalization(512)
            self.conv1_2 = L.Convolution2D(512, 256, (1, 1))
            self.bn1_2 = L.BatchNormalization(256)

            self.conv1_3a = L.Convolution2D(256, out_channels, (1, 1))
            self.conv1_4a = L.Convolution2D(out_channels, 256+128, (1, 1))
            self.conv1_3b = L.Convolution2D(256+128, 256+128, (1, 1))

            self.hg2 = HGBlock(256+128, 512)

            self.conv2_1 = L.Convolution2D(512, 512, (1, 1))
            self.bn2_1 = L.BatchNormalization(512)
            self.conv2_2 = L.Convolution2D(512, 512, (1, 1))
            self.bn2_2 = L.BatchNormalization(512)

            self.conv2_3a = L.Convolution2D(512, out_channels, (1, 1))

    def __call__(self, x):
        # (3, 256, 256) -> (64, 128, 128)
        h = F.relu(self.bn0(self.conv0(x)))
        # (64, 128, 128) -> (128, 64, 64)
        in_1 = _max_pooling_2d(self.res0(h))

        h = self.res1_1(in_1)
        h = self.res1_2(h)
        h = self.res1_3(h)
        # (256, 64, 64)
        h = self.hg1(h)

        # l1 and l2
        h = F.relu(self.bn1_1(self.conv1_1(h)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))

        out_1 = self.conv1_3a(h)

        in_2 = self.conv1_3b(F.concat((in_1, h))) + self.conv1_4a(out_1)

        # l3 and l4
        h = self.hg2(in_2)
        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))

        out_2 = self.conv2_3a(h)

        return out_1, out_2


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=(2, 2), stride=(2, 2))
