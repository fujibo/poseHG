import chainer
from chainer.backends import cuda
from net import StackedHG
from chainercv import utils, transforms

import numpy as np
import cv2

import os


class MPIIVisualizer(object):
    def __init__(self):
        """
        0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip,
        4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax,
        8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow,
        12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
        """
        self.edges = ((0, 11), (11, 12), (12, 7),  # right arm
                      (9, 8), (8, 7),  # head
                      (15, 14), (14, 13), (13, 7),  # left arm
                      (7, 6),  # center
                      (6, 2), (2, 1), (1, 0),  # right leg
                      (6, 3), (3, 4), (4, 5))  # left leg

    def run(self, img, keypoint):
        """

        :param img: np.ndarray shape [C, H, W]
        :param keypoint: np.ndarray shape [16, 2], y, x
        :return:
        """
        img_pose = img.copy()
        for i, edge in enumerate(self.edges):
            point_b = keypoint[edge[0], 1], keypoint[edge[0], 0]
            point_e = keypoint[edge[1], 1], keypoint[edge[1], 0]
            if None in point_b or None in point_e:
                continue

            else:
                point_b = int(point_b[0]), int(point_b[1])
                point_e = int(point_e[0]), int(point_e[1])

            if 0 <= i < 3 or 5 <= i <= 7:  # arm
                color = (0, 0, 255)

            elif 3 <= i < 5:  # center
                color = (0, 255, 0)

            elif i == 8:
                color = (255, 255, 255)

            else:
                color = (255, 0, 0)

            img_pose = cv2.line(img_pose, point_b, point_e, color, 20, 4)

        return img_pose


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', default='')
    parser.add_argument('--image')

    args = parser.parse_args()
    if args.image is None:
        ValueError('args.image should not be None')

    else:
        args.image = os.path.expanduser(args.image)

    model = StackedHG(16)

    if args.model:
        chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    chainer.config.train = False

    img = utils.read_image(args.image)
    img_resized = transforms.resize(img, (256, 256))
    img_resized = img_resized[np.newaxis]

    if args.gpu >= 0:
        img_resized = cuda.to_gpu(img_resized)

    with chainer.no_backprop_mode():
        # (1, 3, 256, 256) -> (1, 16, 64, 64) -> (16, 64, 64)
        _output, output = model(img_resized)
    output = cuda.to_cpu(output.array)[0]

    C, H, W = output.shape

    # (16, 64, 64) -> (16, )
    output = output.reshape(C, -1).argmax(axis=1)
    keypoint = np.unravel_index(output, (H, W))
    keypoint = np.array(keypoint).T
    keypoint = transforms.resize_point(keypoint, (H, W), img.shape[1:])

    img = cv2.imread(args.image)
    visualizer = MPIIVisualizer()
    img_pose = visualizer.run(img, keypoint)

    cv2.imwrite('input.jpg', img)
    cv2.imwrite('output.jpg', img_pose)


if __name__ == '__main__':
    main()
