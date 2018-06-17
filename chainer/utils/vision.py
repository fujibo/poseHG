import numpy as np
import cv2
import chainercv


def zoom(image, ratio=1.0):
    """zoom image
    Args:
        image: C x H x W, BGR or RGB
        ratio: zooming ratio
    Returns:
        zoomed_iamge
    """
    size = image.shape[1]
    center = size // 2
    zoom_mat = np.array([[ratio, 0, (1-ratio)*center],
                         [0, ratio, (1-ratio)*center]])

    zoomed_image = cv2.warpAffine(image.transpose(1, 2, 0), zoom_mat, (size, size))
    return zoomed_image.transpose(2, 0, 1)


def rotate(image, angle):
    """rotate image
    Args:
        image: C x H x W, the order of channel is BGR or RGB
        angle: degrees(-180 - 180)
    Returns:
        rotated_image
    """
    size = image.shape[1]
    center = size // 2
    rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1)
    rotated_image = cv2.warpAffine(image.transpose(1, 2, 0), rot_mat, (size, size))
    return rotated_image.transpose(2, 0, 1)


def flip_heatmap(heatmap, copy=False):
    """flip heatmap following image flipping
    Args:
        heatmap: np.ndarray shape [N, C, H, W]
        copy: bool, default `False`
    Returns:
        heatmap: same shape
    """
    if copy:
        heatmap = heatmap.copy()

    # {'head': [8, 9], 'shoulder': [12, 13], 'elbow': [11, 14], 'wrist': [10, 15], 'hip': [2, 3], 'knee': [1, 4], 'ankle': [0, 5]}
    # correct bias
    flipped_idx = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]
    heatmap = heatmap[:, flipped_idx, :, ::-1]
    return heatmap
