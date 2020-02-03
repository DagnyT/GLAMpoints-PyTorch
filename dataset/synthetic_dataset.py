import numpy as np
import os

from torch.utils.data import Dataset
import torch
import random
import cv2
import glob
import dataset.data_augmentation as data_aug
from .homo_generation import homography_sampling

cv2.setNumThreads(0)

class SyntheticDataset(Dataset):

    def __init__(self, cfg, transform=None, train = True):

        self.cfg = cfg
        self.transform = transform
        self.split = 'train' if train else 'test'
        self.images_path = []

        if self.split == 'train':
            self.root_dir = os.path.join(cfg['INPUT']['TRAIN'])
            self.images_path = open(cfg['INPUT']['TRAIN_FILES']).read().splitlines()

        else:
            self.root_dir = os.path.join(cfg['INPUT']['TEST'])
            self.images_path = open(cfg['INPUT']['TEST_FILES']).read().splitlines()

    def __len__(self):
       return len(self.images_path)

    def __getitem__(self, idx):

        seed = np.random.randint(0,10000)
        random.seed(seed)
        image_full = cv2.imread(os.path.join(self.root_dir, self.images_path[idx]))

        if self.split == 'train':
            random_img_resize = [(1000,1000), (256,256), (512,512),(700,900),(1200,1200), (1200,1500) ,(1200,1400)]
            resize = random.choice(random_img_resize)
            image_full = cv2.resize(image_full,resize)
            image_full = self.center_crop(image_full, (self.cfg['TRAINING']['IMAGE_SIZE_H'], self.cfg['TRAINING']['IMAGE_SIZE_W']))
        else:
            image_full = cv2.resize(image_full, (self.cfg['TEST']['IMAGE_SIZE_H'], self.cfg['TEST']['IMAGE_SIZE_W']))

        image = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)
        h1 = homography_sampling(image.shape, self.cfg['TRAINING']['SAMPLE_HOMOGRAPHY'], seed=seed * (idx + 1))
        image1 = cv2.warpPerspective(np.uint8(image), h1, (image.shape[1], image.shape[0]))
        if image1.max() ==0:
            print(self.images_path[idx])

        image1, image1_preprocessed = self.apply_augmentations(image1, self.cfg['TRAINING']['AUGMENTATION'], seed=seed * (idx + 1))

        h2 = homography_sampling(image.shape, self.cfg['TRAINING']['SAMPLE_HOMOGRAPHY'], seed=seed * (idx + 2))
        image2 = cv2.warpPerspective(np.uint8(image), h2, (image.shape[1], image.shape[0]))
        if image2.max() ==0:
            print(self.images_path[idx])
        image2, image2_preprocessed = self.apply_augmentations(image2, self.cfg['TRAINING']['AUGMENTATION'], seed=seed * (idx + 2))

        H = np.matmul(h2, np.linalg.inv(h1))
        image1 = torch.from_numpy(image1).float().unsqueeze(0)
        image1_preprocessed = torch.from_numpy(image1_preprocessed).float().unsqueeze(0)
        image2 = torch.from_numpy(image2).float().unsqueeze(0)
        image2_preprocessed = torch.from_numpy(image2_preprocessed).float().unsqueeze(0)
        H = torch.from_numpy(H).float()
        return [image1, image1_preprocessed, image2, image2_preprocessed, H]

    def get_random_crop(self, image, crop_height, crop_width):

        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]

        return crop

    def apply_augmentations(self, image, augmentations, seed=None):
        def apply_augmentation(image, aug, augmentations, seed=None):
            '''
            arguments: image - gray image with intensity scale 0-255
                        aug - name of the augmentation to apply
                        seed
            output:
                        image - gray image after augmentation with intensity scale 0-255
            '''
            if aug == 'additive_gaussian_noise':
                image, kp = data_aug.additive_gaussian_noise(image, [], seed=seed,
                                                    std=(augmentations['ADDITIVE_GAUSSIAN_NOISE']['STD_MIN'],
                                                         augmentations['ADDITIVE_GAUSSIAN_NOISE']['STD_MAX']))
            if aug == 'additive_speckle_noise':
                image, kp = data_aug.additive_speckle_noise(image, [], intensity=augmentations['ADDITIVE_SPECKLE_NOISE']['INTENSITY'])
            if aug == 'random_brightness':
                image, kp = data_aug.random_brightness(image, [], seed=seed)
            if aug == 'random_contrast':
                image, kp = data_aug.random_contrast(image, [], seed=seed)
            if aug == 'add_shade':
                image, kp = data_aug.add_shade(image, [], seed=seed)
            if aug == 'motion_blur':
                image, kp = data_aug.motion_blur(image, [], max_ksize=augmentations['MOTION_BLUR']['MAX_KSIZE'])
            if aug == 'gamma_correction':
                # must be applied on image with intensity scale 0-1
                maximum = np.max(image)
                if maximum != 0:
                    image_preprocessed = image / maximum if maximum > 0 else 0
                    random_gamma = random.uniform(augmentations['GAMMA_CORRECTION']['MIN_GAMMA'], \
                                                  augmentations['GAMMA_CORRECTION']['MAX_GAMMA'])
                    image_preprocessed = image_preprocessed ** random_gamma
                    image = image_preprocessed * maximum
            if aug == 'opposite':
                # must be applied on image with intensity scale 0-1
                maximum = np.max(image)
                if maximum != 0:
                    image_preprocessed = image / maximum if maximum > 0 else 0
                    image_preprocessed = 1 - image_preprocessed
                    image = image_preprocessed * maximum
            if aug == 'no_aug':
                pass
            return image

        random.seed(seed)
        list_of_augmentations = augmentations['AUGMENTATION_LIST']
        index = random.sample(range(len(list_of_augmentations)), 3)
        for i in index:
            aug = list_of_augmentations[i]
            image = apply_augmentation(image, aug, augmentations, seed)

        image_preprocessed = image / (np.max(image) + 0.000001)
        return image, image_preprocessed

    def center_crop(self, img, size):
        """
        Get the center crop of the input image
        Args:
            img: input image [HxWx3]
            size: size of the center crop (tuple)
        Output:
            img_pad: center crop
            x, y: coordinates of the crop
        """

        if not isinstance(size, tuple):
            size = (size, size)

        img = img.copy()
        h, w = img.shape[:2]

        pad_w = 0
        pad_h = 0
        if w < size[1]:
            pad_w = np.uint16((size[1] - w) / 2)
        if h < size[0]:
            pad_h = np.uint16((size[0] - h) / 2)
        img_pad = cv2.copyMakeBorder(img,
                                     pad_h,
                                     pad_h,
                                     pad_w,
                                     pad_w,
                                     cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])
        x1 = w // 2 - size[1] // 2
        y1 = h // 2 - size[0] // 2

        img_pad = img_pad[y1:y1 + size[0], x1:x1 + size[1], :]

        return img_pad