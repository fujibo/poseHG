import chainer
from chainer.backends import cuda
from net import StackedHG
from chainercv import utils, transforms

import numpy as np
import cv2

import os

from snap2model import snap2model_trainer
from utils.demo_helper import MPIIVisualizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', default='', help='if not specified, you download and use a pre-trained model.')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--image', type=str)

    args = parser.parse_args()
    if not args.image:
        ValueError('args.image should be specified.')

    else:
        args.image = os.path.expanduser(args.image)

    model = StackedHG(16)

    if args.model:
        chainer.serializers.load_npz(args.model, model)

    elif args.snapshot:
        chainer.serializers.load_npz(snap2model_trainer(args.snapshot), model)

    else:
        # use pre-trained model
        from google_drive_downloader import GoogleDriveDownloader as gdd
        
        model_path = './models/model_2018_05_22.npz'
        if not os.path.exists(model_path):
            gdd.download_file_from_google_drive(file_id='1rZZJRpqQKkncn30Igtk8KirgR96QlCFO', dest_path=model_path)

        chainer.serializers.load_npz(model_path, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    chainer.config.train = False

    img = utils.read_image(args.image)
    img = img / 255.
    img = img.astype(np.float32)

    # expected properties
    # - A person is in the center of the image
    # - the height of this image == 1.25 * a person's scale (= height)
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
