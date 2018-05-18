import chainer
from chainer.dataset import DatasetMixin
from chainercv import transforms, utils

from utils import vision

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

    self.keypoints
        0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip,
        4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax,
        8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow,
        12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
    """
    def __init__(self, root='~/data/MPII', split='train'):
        super(MPIIDataset, self).__init__()
        self._root = os.path.expanduser(root)
        self._split = split

        fname = f'{self._root}/annotations/mpii_human_pose_v1_u12_1.mat'
        paths, keypoints, head_sizes, centers, scales =  _read_mpii_annots(fname, split)

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

        img, point, idx, shape = preprocess(img, keypoint, center, scale)

        if self._split == 'train':
            label = point2heatmap(point, idx, shape)
            img = augment_data(img, label)

            return img, label, idx

        # 'val' or 'test'
        else:
            return img, point, idx, self.head_sizes[i], shape

    def to_hdf5(self):
        """convert to hdf5 file format and save
        """
        import h5py

        points_x = list()
        points_y = list()
        visibles = list()
        angles = list()

        # for each person, get keypoint
        for keypoint in self.keypoints:
            xx = np.array(keypoint['x']).astype(np.float32)
            yy = np.array(keypoint['y']).astype(np.float32)
            visible = np.array(keypoint['visible']).astype(np.float32)
            keypoint['visible']

            points_x.append(xx)
            points_y.append(yy)
            visibles.append(visible)

            point_pelvis = xx[6], yy[6]
            point_thorax = xx[7], yy[7]

            # to imagenary num
            p_pel = point_pelvis[0] + 1j * point_pelvis[1]
            p_thr = point_thorax[0] + 1j * point_thorax[1]

            angle = 90 + np.angle(p_thr - p_pel, deg=True)
            angles.append(angle)


        # 2 x dsize x 16 -> dsize x 16 x 2 (None -> np.nan)
        points = np.array([points_x, points_y]).transpose(1, 2, 0)
        # np.nan -> 0
        points = np.nan_to_num(points)

        visibles = np.array(visibles)

        angles = np.array(angles)
        # np.nan -> 0
        points = np.nan_to_num(points)

        data_h5 = dict()
        data_h5['normalize'] = np.array(self.head_sizes)
        data_h5['center'] = np.array(self.centers)
        data_h5['scale'] = np.array(self.scales) / 200
        data_h5['part'] = points
        data_h5['imgname'] = np.array(self.paths, dtype=bytes)
        data_h5['visible'] = visibles

        # https://github.com/umich-vl/pose-hg-train/blob/master/src/misc/mpii.py
        # Get angle from pelvis (6) to thorax (7)
        data_h5['torsoangle'] = angles

        with h5py.File(f'mpii-{self._split}.h5', mode='w') as f:
            f.update(data_h5)

        with open(f'mpii-{self._split}_images.txt', 'w') as f:
            for path in self.paths:
                f.write(f'{path}\n')


def _read_mpii_annots(fname, split):
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

            # pose-hg-train/src/util/dataset/mpii.lua Dataset:getPartInfo
            # --Small adjustment so cropping is less likely to take feet out
            center = [an.objpos.x, an.objpos.y + 15 * an.scale]
            scale = an.scale * 200 * 1.25

            paths.append(path)
            keypoints.append(keypoint)
            centers.append(center)
            scales.append(scale)

            # in validation, we use only a part of annotations
            if split == 'val':
                break

    head_size = np.array(head_size)
    print(len(paths))
    assert len(paths) == len(keypoints)

    return paths, keypoints, head_size, centers, scales


def point2heatmap(points, indices, input_shape):
    """convert keypoint of a person to heatmap
    Args:
        points: np.ndarray, shape [16, 2]
        indices: np.ndarray, available point or not shape [16, ]
        input_shape: tuple, input shape of an image of the person

    Returns:
        heatmap: shape [16, 64, 64]
    """
    points = transforms.resize_point(points, input_shape, (64, 64))

    # pose-hg-train/src/utils/img.lua drawGaussian
    # pose-hg-train/src/utils/pose.lua generateSample
    heatmap = np.zeros((16, 64, 64), dtype=np.float32)
    sigma = 1.0
    for i, (available, point) in enumerate(zip(indices, points)):
        if available:
            coordinates = np.array(np.meshgrid(range(64), range(64)))
            diff = coordinates - point[:, None, None]
            dist = np.sum(diff ** 2, axis=0)
            heatmap[i] = np.exp(-dist / (2 * sigma**2))

    return heatmap


def augment_data(img, heatmap=None):
    """data augmentation on image
    Args:
        img: shape [C, H, W]
        heatmap: shape [16, 64, 64] or None
    Returns:
        img: shape [C, H, W]
        heatmap: shape [16, 64, 64] or None
    """
    # rotate [-30, 30]
    angle = np.random.uniform(-30, 30)
    img = vision.rotate(img, angle)

    if heatmap is not None:
        heatmap = vision.rotate(heatmap, angle)

    # scale [0.75, 1.25]
    scale_zoom = np.random.uniform(0.75, 1.25)
    img = vision.zoom(img, scale_zoom)
    if heatmap is not None:
        heatmap = vision.zoom(heatmap, scale_zoom)

    # NOTE
    # They use other data augmentations in pose-hg-train/src/pose.lua
    # though they didn't report it in their paper.
    # I decided to utilize them.

    # flip
    img, param = transforms.random_flip(img, x_random=True, return_param=True)

    if param['x_flip'] and heatmap is not None:
        heatmap = vision.flip_heatmap(heatmap[np.newaxis])[0]

    # color
    weight = np.random.uniform(0.6, 1.4, size=3)[:, None, None]
    img = weight * img
    img = np.clip(img, 0.0, 1.0)

    img = img.astype(np.float32)
    return img


def preprocess(img, keypoint, center, scale):
    """preprocess image and keypoint
    Args:
        img: CxHxW
        keypoint: dict {'x': [None]*16, 'y': [None]*16, 'visible': [None]*16}
        center: [x, y] center of a person
        scale: float, size of a person
    Returns:
        img: shape [C, 256, 256], (not copied)
        points: shape [16, 2]
        indices: index available
        image_shape:
    """
    # [0, 255] -> [0, 1]
    img = img / 255.
    img = img.astype(np.float32)

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

    image_shape = ymax - ymin, xmax - xmin

    return img, points, indices, image_shape


if __name__ == '__main__':
    dataset = MPIIDataset(split='val')
    dataset.to_hdf5()
    # dataset.get_example(15)
