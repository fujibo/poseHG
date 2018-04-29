import chainer
from chainer.backends import cuda
from net import StackedHG
from chainercv import transforms

import numpy as np

from dataset import MPIIDataset
import os

from chainer.dataset import concat_examples


def pckh_score(keypoint_true, keypoint_pred, idx, scale):
    """evaluation of keypoint

    :param keypoint_true: [N, 16, 2]
    :param keypoint_pred: points shape [N, 16, 2]
    :param idx: [N, 16]
    :param scale: [N, ]
    :return:
        correct: int
        size: int
    """
    # (N, 16, 2) -> (N, 16)
    dist = np.sqrt(np.sum((keypoint_true - keypoint_pred) ** 2, axis=2))

    # normalized by the head size
    dist = dist / scale
    n_joints = idx.shape[1]
    correct = [np.sum(dist[idx[:, i], i] <= 0.5) for i in range(n_joints)]
    correct = np.array(correct)
    size = np.sum(idx, axis=0)

    return correct, size


def evaluate(model, dataset, device=-1):

    data_iter = chainer.iterators.MultithreadIterator(dataset, 100, repeat=False, shuffle=False)

    corrects = list()
    counts = list()

    for batch in data_iter:
        img, label, idx, scale = concat_examples(batch)

        shape = list()
        img_resized = list()
        for i in range(len(batch)):
            shape.append(img.shape)
            img_ = transforms.resize(img[i], (256, 256))
            img_resized.append(img_)

        img_resized = np.array(img_resized)
        if device >= 0:
            img_resized = cuda.to_gpu(img_resized)

        # (N, 3, 256, 256) -> (N, 16, 64, 64)
        _output, output = model(img_resized)
        output = cuda.to_cpu(output.array)

        C, H, W = output.shape

        # (16, 64, 64) -> (16, )
        output = output.reshape(C, -1).argmax(axis=1)
        keypoint = np.unravel_index(output, (H, W))
        keypoint = np.array(keypoint).T
        keypoint = transforms.resize_point(keypoint, (H, W), img.shape[1:])

        correct, count = pckh_score(label, keypoint, idx, scale)
        corrects.append(correct)
        counts.append(count)

    corrects = np.sum(corrects, axis=0)
    counts = np.sum(counts, axis=0)
    # Head, Shoulder, Elbow, Wrist, Hip, Knee, Ankle
    joints = {'head': (9, ), 'shoulder': (12, 13), 'elbow': (11, 14),
              'wrist': (10, 15), 'hip': (2, 3), 'knee': (1, 4), 'ankle': (0, 5)}

    scores = dict()
    for key, value in joints.items():
        score = corrects[value].sum() / counts[value].sum()
        scores.update({key: score})
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', default='')

    args = parser.parse_args()

    model = StackedHG(16)

    if args.model:
        chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    chainer.config.train = False

    dataset = MPIIDataset(split='val')
    scores = evaluate(model, dataset, args.gpu)
    print(scores)



if __name__ == '__main__':
    main()
