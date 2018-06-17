import cv2


class MPIIVisualizer(object):
    def __init__(self):
        """
        0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip,
        4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax,
        8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow,
        12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
        """
        self.edges = ((10, 11), (11, 12), (12, 7),  # right arm
                      (9, 8), (8, 7),  # head
                      (15, 14), (14, 13), (13, 7),  # left arm
                      (7, 6),  # center
                      (6, 2), (2, 1), (1, 0),  # right leg
                      (6, 3), (3, 4), (4, 5))  # left leg

    def run(self, img, keypoint):
        """

        :param img: np.ndarray shape [H, W, C] (by using cv2)
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
