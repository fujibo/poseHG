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
    """
    def __init__(self, root='~/data/MPII', split='train'):
        super(MPIIDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self._split = split

        fname = f'{self._root}/annotations/mpii_human_pose_v1_u12_1.mat'
        paths, annots =  read_mpii_annots(fname, split)

        self.paths = paths
        self.keypoints = annots

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):

        img_path = f'{self._root}/images/{self.paths[i]}'
        img = utils.read_image(img_path)

        keypoint = self.keypoints[i]
        img, label, idx = preprocess(img, keypoint)

        return img, label, idx


def read_mpii_annots(fname, split):
    """
    Args:
        fname: str
        split: str, 'train' or 'val'
    Returns:
        paths: list of str
        annotations: list of dict
    """
    # TODO: implimentation of train, val split

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
    annotations = list()
    for data_idx in np.where(arr.img_train == 1)[0]:
        annot = arr.annolist[data_idx]
        path = annot.image.name

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

            paths.append(path)
            annotations.append(keypoint)

    print(len(paths))
    assert len(paths) == len(annotations)

    return paths, annotations


def preprocess(img, keypoint):
    """preprocess image and keypoint
    Args:
        img: CxHxW
        keypoint: dict {'x': [None]*16, 'y': [None]*16, 'visible': [None]*16}
    Returns:
        img: shape Cx256x256, (not copied)
        heatmap: shape 16x64x64
        indices: index available
    """
    # image processing
    shape = img.shape[1:]
    img = transforms.resize(img, (256, 256))

    if chainer.config.train:
        # rotate [-30, 30]
        angle = np.random.uniform(-30, 30)
        img = vision.rotate(img, angle)

        # scale [0.75, 1.25]
        scale = np.random.uniform(0.75, 1.25)
        img = vision.zoom(img, scale)

    # key point processing
    heatmap = np.zeros((16, 64, 64), dtype=np.float32)
    indices = list()
    for i, (x, y) in enumerate(zip(keypoint['x'], keypoint['y'])):
        point = np.array((y, x))

        if None in (x, y):
            continue
        else:
            point_resized = transforms.resize_point(point[np.newaxis], shape, (64, 64))
            y, x = point_resized[0].astype(np.int64)

            heatmap[i, y, x] = 1
            heatmap[i] = gaussian_filter(heatmap[i], sigma=1)
            indices.append(i)

    indices = np.array(indices)

    return img, heatmap, indices
