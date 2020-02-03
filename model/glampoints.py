import numpy as np
import cv2
from skimage.feature import peak_local_max
from model.model import Unet_model_4

def sift(SIFT, image, kp_before):

    kp, des = SIFT.compute(image, kp_before)
    if des is not None:
        eps = 1e-7
        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)
    return kp, des

def non_max_suppression(image, size_filter, proba):

    non_max = peak_local_max(image, min_distance=size_filter, threshold_abs=proba, \
                             exclude_border=True, indices=False)
    kp = np.where(non_max > 0)
    if len(kp[0]) != 0:
        for i in range(len(kp[0])):

            window = non_max[kp[0][i] - size_filter:kp[0][i] + (size_filter + 1), \
                     kp[1][i] - size_filter:kp[1][i] + (size_filter + 1)]
            if np.sum(window) > 1:
                window[:, :] = 0
    return non_max

class GLAMpointsInference:

    def __init__(self, cfg):

        self.path_weights = str(cfg['INPUT']['WEIGHTS'])
        self.nms = int(cfg['INPUT']['NMS'])
        self.min_prob = float(cfg['INPUT']['MIN_PROB'])
        self.kpmap = Unet_model_4(cfg['INPUT']['INPUT_LAYER'])

    def find_and_describe_keypoints(self, image):

        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_norm = image / np.max(image)
        image_to_feed = np.zeros((1, image_norm.shape[0], image_norm.shape[1], 1))
        image_to_feed[0, :image_norm.shape[0], :image_norm.shape[1], 0] = image_norm

        # write here
        kp_map = self.kpmap(image_to_feed)
        kp_map_nonmax = non_max_suppression(kp_map[0, :, :, 0], self.nms, self.min_prob)

        keypoints_map = np.where(kp_map_nonmax > 0)
        kp_array = np.array([keypoints_map[1], keypoints_map[0]]).T
        kp_cv2 = [cv2.KeyPoint(kp_array[i, 0], kp_array[i, 1], 10) for i in range(len(kp_array))]
        kp, des = sift(np.uint8(image), kp_cv2)
        kp = np.array([m.pt for m in kp], dtype=np.int32)
        return kp, des

class SIFT_noorientation:

    def __init__(self, **kwargs):

        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(kwargs['nfeatures']),
                                                contrastThreshold=float(kwargs['contrastThreshold']),
                                                edgeThreshold= float(kwargs['edgeThreshold']),
                                                sigma=float(kwargs['sigma']))

    def find_and_describe_keypoints(self, image):
        eps = 1e-7
        kp = self.sift.detect(image, None)
        kp = np.int32([kp[i].pt for i in range(len(kp))])
        kp = [cv2.KeyPoint(kp[i, 0], kp[i, 1], 10) for i in range(len(kp))]
        kp, des = self.sift.compute(image, kp)
        if des is not None:
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)
        kp = np.array([kp[i].pt for i in range(len(kp))])
        return kp, des


class SIFT:
    def __init__(self, **kwargs):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(kwargs['nfeatures']),
                                                contrastThreshold=float(kwargs['contrastThreshold']),
                                                edgeThreshold= float(kwargs['edgeThreshold']),
                                                sigma=float(kwargs['sigma']))

    def find_and_describe_keypoints(self, image):
        eps = 1e-7
        kp = self.sift.detect(image, None)
        kp, des = self.sift.compute(image, kp)
        if des is not None:
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)
        kp = np.array([kp[i].pt for i in range(len(kp))])
        return kp, des