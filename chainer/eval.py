import chainer
from chainer.backends import cuda
from net import StackedHG
from chainercv import transforms

import numpy as np

from dataset import MPIIDataset
import os

from chainer.dataset import concat_examples
from snap2model import snap2model_trainer


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
    dist = dist / scale[:, np.newaxis]
    n_joints = idx.shape[1]
    correct = [np.sum(dist[idx[:, i], i] <= 0.5) for i in range(n_joints)]
    correct = np.array(correct)
    size = np.sum(idx, axis=0)

    return correct, size


def evaluate(model, dataset, device=-1, flip=False):

    batch_size = 50
    data_iter = chainer.iterators.MultithreadIterator(dataset, batch_size, repeat=False, shuffle=False)

    corrects = list()
    counts = list()

    for it, batch in enumerate(data_iter):
        # print progress
        print(f'{batch_size*it:04d} / {len(dataset):04d}', end='\r')

        img, label, idx, scale, shape = concat_examples(batch)
        N, C, H, W = img.shape

        if flip:
            img = np.array((img, img[:, :, :, ::-1]))
            img = img.reshape(N*2, C, H, W)

        if device >= 0:
            img = cuda.to_gpu(img)

        with chainer.no_backprop_mode():
            # (N, 3, 256, 256) -> (N, 16, 64, 64)

            _output, output = model(img)

        output = output.array

        if flip:
            # {'head': [8, 9], 'shoulder': [12, 13], 'elbow': [11, 14], 'wrist': [10, 15], 'hip': [2, 3], 'knee': [1, 4], 'ankle': [0, 5]}
            # correct bias
            flipped_idx = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

            output = output.reshape((2, N, ) + output.shape[1:])
            output = (output[0] + output[1, :, flipped_idx, :, ::-1].transpose(1, 0, 2, 3)) / 2

        N, C, H, W = output.shape

        keypoints = list()
        # (N, 16, 64, 64) -> (N, 16, 2)
        for i in range(N):
            # (16, 64, 64) -> (16, -1)
            out_reshaped = output[i].reshape(C, -1).argmax(axis=1)
            out_reshaped = cuda.to_cpu(out_reshaped)
            keypoint = np.unravel_index(out_reshaped, (H, W))
            # (2, 16) -> (16, 2)
            keypoint = np.array(keypoint).T
            keypoint = transforms.resize_point(keypoint, (H, W), shape[i])
            keypoints.append(keypoint)

        else:
            keypoints = np.array(keypoints)

        correct, count = pckh_score(label, keypoints, idx, scale)
        corrects.append(correct)
        counts.append(count)

    print()
    corrects = np.sum(corrects, axis=0)
    counts = np.sum(counts, axis=0)
    # Head, Shoulder, Elbow, Wrist, Hip, Knee, Ankle
    joints = {'head': [8, 9], 'shoulder': [12, 13], 'elbow': [11, 14],
              'wrist': [10, 15], 'hip': [2, 3], 'knee': [1, 4], 'ankle': [0, 5]}

    scores = dict()
    for key, value in joints.items():
        score = corrects[value].sum() / counts[value].sum()
        scores.update({key: score})
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--flip', action='store_true', help='test time augment')
    parser.add_argument('--model', default='')
    parser.add_argument('--snapshot', default='')


    args = parser.parse_args()

    model = StackedHG(16)

    if args.model:
        chainer.serializers.load_npz(args.model, model)

    elif args.snapshot:
        chainer.serializers.load_npz(snap2model_trainer(args.snapshot), model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = MPIIDataset(split='val')
    with chainer.using_config('train', False):
        scores = evaluate(model, dataset, args.gpu, args.flip)
    print(scores)



if __name__ == '__main__':
    main()
