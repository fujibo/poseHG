import chainer
from chainer.backends import cuda
from chainercv import utils, transforms
from chainercv.links import SSD512

import cv2
import numpy as np

import os

from demo import MPIIVisualizer
from net import StackedHG

from snap2model import snap2model_trainer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', default='')
    parser.add_argument('--snapshot', default='')
    parser.add_argument('--image', type=str)

    args = parser.parse_args()
    if not args.image:
        ValueError('args.image should be specified.')

    else:
        args.image = os.path.expanduser(args.image)

    detector = SSD512(pretrained_model='voc0712')
    model = StackedHG(16)

    if args.model:
        chainer.serializers.load_npz(args.model, model)

    elif args.snapshot:
        chainer.serializers.load_npz(snap2model_trainer(args.snapshot), model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        detector.to_gpu()
        model.to_gpu()

    chainer.config.train = False

    img = utils.read_image(args.image)

    # detect persons
    bboxes, labels, scores = detector.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    # expand bboxes and crop the image
    img = img / 255.
    img = img.astype(np.float32)

    img_persons = list()
    bbox_persons = list()
    for ymin, xmin, ymax, xmax in bbox:
        scale = ymax - ymin

        # this is for ankle (also used in training with mpii dataset)
        offset = 15 / 200 * scale
        center = (xmin + xmax) / 2, (ymin + ymax) / 2 + offset

        # this is for ankle (also used in training with mpii dataset)
        scale *= 1.25

        xmin, xmax = center[0] - scale / 2, center[0] + scale / 2
        ymin, ymax = center[1] - scale / 2, center[1] + scale / 2

        # truncate
        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(img.shape[2], xmax))
        ymax = int(min(img.shape[1], ymax))

        # croping
        img_person = img[:, ymin:ymax, xmin:xmax]
        img_person = transforms.resize(img_person, (256, 256))

        img_persons.append(img_person)
        bbox_persons.append((ymin, xmin, ymax, xmax))

    img_persons = np.array(img_persons)
    bbox_persons = np.array(bbox_persons)

    utils.write_image(utils.tile_images(img_persons, n_col=2), 'tiled.jpg')

    # estimate poses
    if args.gpu >= 0:
        img_persons = cuda.to_gpu(img_persons)

    with chainer.no_backprop_mode():
        # (R, 3, 256, 256) -> (R, 16, 64, 64) -> (16, 64, 64)
        _outputs, outputs = model(img_persons)
    outputs = cuda.to_cpu(outputs.array)

    R, C, H, W = outputs.shape

    # heatmap to keypoint
    # R, C, H, W -> R, C, 2
    keypoints = list()
    for output in outputs:
        # (16, 64, 64) -> (16, )
        output = output.reshape(C, -1).argmax(axis=1)
        keypoint = np.unravel_index(output, (H, W))
        keypoint = np.array(keypoint).T
        keypoints.append(keypoint)

    # keypoint (local) to keypoint (global)
    keypoint_persons = list()
    for keypoint, bbox_person in zip(keypoints, bbox_persons):
        ymin, xmin, ymax, xmax = bbox_person
        keypoint = transforms.resize_point(keypoint, (H, W), (ymax-ymin, xmax-xmin))
        keypoint_person = keypoint + np.array((ymin, xmin))
        keypoint_persons.append(keypoint_person)

    # visualize
    img = cv2.imread(args.image)
    visualizer = MPIIVisualizer()

    img_pose = img.copy()
    for keypoint_person, bbox_person in zip(keypoint_persons, bbox_persons):
        ymin, xmin, ymax, xmax = bbox_person

        img_pose = visualizer.run(img_pose, keypoint_person)
        img_pose = cv2.rectangle(img_pose, (xmin, ymin), (xmax, ymax), (0, 255, 255), 50)

    cv2.imwrite('input.jpg', img)
    cv2.imwrite('output.jpg', img_pose)


if __name__ == '__main__':
    main()
