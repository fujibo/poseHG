import unittest
import numpy as np

from net import StackedHG
from dataset import MPIIDataset

class NetTest(unittest.TestCase):

    def setUp(self):
        self.out_channels = 16
        self.model = StackedHG(self.out_channels)
        self.image = np.random.rand(1, 3, 256, 256).astype(np.float32)

    def output_shape(self):
        out1, out2 = self.model(self.image)
        self.assertEqual(out1.shape, (1, self.out_channels, 64, 64), 'unexpected shape')
        self.assertEqual(out2.shape, (1, self.out_channels, 64, 64), 'unexpected shape')


class DatasetTest(unittest.TestCase):

    def setUp(self):
        print('loading...')
        self.dataset = MPIIDataset()
        print('loaded')

    def get_example_test(self):
        dataset = MPIIDataset()
        out = dataset.get_example(12)
        # dataset.get_example(62)
        print(out)


if __name__ == '__main__':
    unittest.main()
