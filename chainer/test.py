import chainer
from dataset import MPIIDataset

import h5py
import matplotlib.pyplot as plt

import numpy as np
from net import StackedHG

import unittest


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

    def get_example_train(self):
        dataset = MPIIDataset(split='train')
        img, heatmap, idx = dataset.get_example(15)
        plt.imshow(img.transpose(1, 2, 0).astype(np.uint8))
        plt.show()
        plt.imshow(heatmap[0])
        plt.gray()

    def get_example_test(self):
        dataset = MPIIDataset(split='val')
        img, keypoint, idx, head_size, shape = dataset.get_example(15)
        from demo import MPIIVisualizer
        visualizer = MPIIVisualizer()
        img_pose = visualizer.run(img, keypoint)
        plt.imshow(img_pose.transpose(1, 2, 0).astype(np.uint8))
        plt.show()

class MetricsTest(unittest.TestCase):

    def setUp(self):
        self.dataset = MPIIDataset(split='val')
        self.preds = h5py.File('../../pose-hg-demo/preds/valid-example.h5')['preds'].value

    def evaluate(self):
        pass
        # assert(len(self.dataset) == self.preds.shape[0])
        # for i in range(self.dataset):
        #     img, label, idx, scale, shape = self.dataset[i]
        #     label, self.preds[i]


if __name__ == '__main__':
    unittest.main()
