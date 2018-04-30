import chainer
from chainer.dataset import DatasetMixin
from chainercv import transforms, utils

from libs import vision

import numpy as np
import  pickle

import os
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter


class MPIIDataset(DatasetMixin):
    """MPII dataset
    get_example returns img, heatmap, idx available for training

    Args:
        split: str 'train', 'val'

    Returns:
        dataset
    """
    def __init__(self, root='~/data/MPII', split='train'):
        super(MPIIDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self._split = split

        fname = f'{self._root}/annotations/mpii_human_pose_v1_u12_1.mat'
        paths, keypoints, head_sizes, centers, scales =  read_mpii_annots(fname, split)

        self.paths = paths
        self.keypoints = keypoints
        self.head_sizes = head_sizes

        self.centers = centers
        self.scales = scales

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        img_path = f'{self._root}/images/{self.paths[i]}'
        img = utils.read_image(img_path)

        keypoint = self.keypoints[i]
        center = self.centers[i]
        scale = self.scales[i]

        if self._split == 'train':
            img, label, idx = preprocess(img, keypoint, center, scale, self._split)
            return img, label, idx

        # 'val' or 'test'
        else:
            img, point, idx, shape = preprocess(img, keypoint, center, scale, self._split)
            return img, point, idx, self.head_sizes[i], shape


def read_mpii_annots(fname, split):
    """
    Args:
        fname: str
        split: str, 'train' or 'val'
    Returns:
        paths: list of str
        keypoints: list of dict
        head_size: list of float
    """
    with open('./valid_images.txt') as f:
        text = f.read()
        validation = text.split('\n')[:-1]

    path_cache = 'mpii.pickle'
    if os.path.exists(path_cache):
        with open(path_cache, 'rb') as f:
            arr = pickle.load(f)

    else:
        mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
        arr = mat['RELEASE']
        with open(path_cache, 'wb') as f:
            pickle.dump(arr, f)

    paths = list()
    keypoints = list()
    head_size = list()

    centers = list()
    scales = list()

    for data_idx in np.where(arr.img_train == 1)[0]:
        annot = arr.annolist[data_idx]
        path = annot.image.name

        if split == 'train':
            if path in validation:
                continue

        if split == 'val':
            if not path in validation:
                continue

        # there are some people
        if type(annot.annorect) is np.ndarray or type(annot.annorect) is list:
            pass

        # there is only 1 person
        else:
            annot.annorect = [annot.annorect]

        # each person
        for an in annot.annorect:
            if not hasattr(an, 'annopoints'):
                continue

            if not an.annopoints:
                continue

            if type(an.annopoints.point) is not list and type(an.annopoints.point) is not np.ndarray:
                continue

            keypoint = {'x': [None]*16, 'y': [None]*16, 'visible': [None]*16}
            for point in an.annopoints.point:
                keypoint['x'][point.id] = point.x
                keypoint['y'][point.id] = point.y
                keypoint['visible'][point.id] = bool(point.is_visible)

            # http://human-pose.mpi-inf.mpg.de/results/mpii_human_pose/evalMPII.zip
            SC_BIAS = 0.8 * 0.75
            head_size.append(SC_BIAS * np.linalg.norm([an.x2 - an.x1, an.y2 - an.y1]))

            center = [(an.x1 + an.x2)/2, (an.y1 + an.y2)/2]
            scale = an.scale * 200

            paths.append(path)
            keypoints.append(keypoint)
            centers.append(center)
            scales.append(scale)

    head_size = np.array(head_size)
    print(len(paths))
    assert len(paths) == len(keypoints)

    return paths, keypoints, head_size, centers, scales


def preprocess(img, keypoint, center, scale, mode='train'):
    """preprocess image and keypoint
    Args:
        img: CxHxW
        keypoint: dict {'x': [None]*16, 'y': [None]*16, 'visible': [None]*16}
        center: [x, y] center of a person
        scale: float, size of a person
        mode: 'train', 'val', 'test'
    Returns:
        img: shape Cx256x256, (not copied)
        heatmap: shape 16x64x64
        indices: index available
    """
    # image processing
    shape = img.shape

    xmin, xmax = center[0] - scale / 2, center[0] + scale / 2
    ymin, ymax = center[1] - scale / 2, center[1] + scale / 2

    # truncate
    xmin = int(max(0, xmin))
    ymin = int(max(0, ymin))
    xmax = int(min(shape[2], xmax))
    ymax = int(min(shape[1], ymax))

    # croping
    img = img[:, ymin:ymax, xmin:xmax]
    img = transforms.resize(img, (256, 256))

    # key point processing
    indices = list()
    points = list()
    for i, (x, y) in enumerate(zip(keypoint['x'], keypoint['y'])):
        if None in (x, y):
            point = (-1000, -1000)
            indices.append(False)

        else:
            point = (y - ymin, x - xmin)
            indices.append(True)

        points.append(point)

    points = np.array(points)
    indices = np.array(indices)

    if mode == 'train':
        points_resized = transforms.resize_point(points, shape[1:], (64, 64))
        points_resized = points_resized.astype(np.int64)
        points_resized[points_resized < 0] = 0
        points_resized[points_resized > 63] = 63

        heatmap = np.zeros((16, 64, 64), dtype=np.float32)
        for i, (available, point) in enumerate(zip(indices, points_resized)):
            if available:
                y, x = point
                heatmap[i, y, x] = 1
                heatmap[i] = gaussian_filter(heatmap[i], sigma=1)

        # data augmetation
        # rotate [-30, 30]
        angle = np.random.uniform(-30, 30)
        img = vision.rotate(img, angle)
        heatmap = vision.rotate(heatmap, angle)

        # scale [0.75, 1.25]
        scale_zoom = np.random.uniform(0.75, 1.25)
        img = vision.zoom(img, scale_zoom)
        heatmap = vision.zoom(heatmap, scale_zoom)


        return img, heatmap, indices

    else:
        return img, points, indices, (ymax-ymin, xmax-xmin)


if __name__ == '__main__':
    dataset = MPIIDataset()
    dataset.get_example(15)
