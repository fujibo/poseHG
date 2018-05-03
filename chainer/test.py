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
        # NOTE: their validation and our validation is not identical.
        self.preds = h5py.File('../../pose-hg-demo/preds/valid-example.h5')['preds'].value
        annot = h5py.File('../../pose-hg-demo/annot/valid.h5')

        self.annot = dict()
        for key, value in annot.items():
            self.annot.update({key: value.value})

    def evaluate(self):
        points = list()
        indices = list()
        scales = list()

        for i in range(self.preds.shape[0]):
            keypoint = self.annot['part'][i]
            scale = self.annot['normalize'][i]

            point = np.zeros((16, 2), dtype=np.float)
            idx = np.zeros(16, dtype=np.bool)
            for i, (x, y) in enumerate(keypoint):
                if np.all(np.isclose((x, y), (0, 0))):
                    # NOTE: should not be used
                    point[i] = (-1, -1)
                    idx[i] = False

                else:
                    point[i] = (y, x)
                    idx[i] = True

            points.append(point)
            indices.append(idx)
            scales.append(scale)

        from eval import pckh_score
        corrects, counts = pckh_score(points, self.preds[:, :, ::-1], indices, scales)

        # ignores 6, 7, 8
        joints = {'head': [8, 9], 'shoulder': [12, 13], 'elbow': [11, 14],
                  'wrist': [10, 15], 'hip': [2, 3], 'knee': [1, 4], 'ankle': [0, 5]}

        scores = dict()
        for key, value in joints.items():
            score = corrects[value].sum() / counts[value].sum()
            scores.update({key: score})

        # this outputs same score with pose-hg-demo/main.lua
        print(scores)

        # assert(len(self.dataset) == self.preds.shape[0])
        # for i in range(self.dataset):
        #     img, label, idx, scale, shape = self.dataset[i]
        #     label, self.preds[i]


if __name__ == '__main__':
    unittest.main()
