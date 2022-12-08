# Necessary packages
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt
import cv2
import skimage.exposure
from albumentations.core.transforms_interface import ImageOnlyTransform

class Blobs(ImageOnlyTransform):
    """
    This class is defined to implement random blob augmentation. It adds
    random blobs, which are similar to ellipses, at random positions in
    an image. Doing so increases the validation performance.
    """
    def apply(self, img, **params):
        return add_blobs(img)


def add_blobs(img):
    """
    Takes the image and adds random blobs to the random positions of image.
    :param img: image with segmentation mask (prediction of PCL or any segmenter)
    :return: image containing random blobs
    """
    height, width = img.shape[:2]
    img = img.astype(np.uint8)
    # create random noise image
    noise = np.random.randint(0, 4, (height, width), np.uint8)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    mask = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # add mask to input
    result = cv2.add(img, mask)-1
    return result


class ACDCseg(Dataset):

    """
    This dataset class is defined for training of our DPCL algorithm.
    It consists of the predictions with spatial augmentations made by PCL.
    """

    def __init__(self, keys, img_dir='seg_masks', label_dir='ground_truths', mri_dir='mri_images', transform=None, num_classes=4):

        self.img_files = [os.path.join(img_dir, str(file_name) + '.npy') for file_name in keys] # prediction mask file names
        self.mask_files = [os.path.join(label_dir, str(file_name) + '.npy') for file_name in keys] # ground truth file names
        self.mri_files = [os.path.join(mri_dir, str(file_name) + '.npy') for file_name in keys] # mri file names
        self.transform = transform # augmentations
        self.num_classes = num_classes # right ventricle, left ventricle, myocardium, background

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load the prediction, ground truth and the original mri image (for display purposes later)
        img, mask, mri = np.load(self.img_files[index]), np.load(self.mask_files[index]), np.load(self.mri_files[index])
        # augmentation on images
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']

        img = self.img_processor(img - 1) # -1 is because class labels start from 1 instead of 0

        return img, mask - 1, mri

    def img_processor(self, img):
        """
        This functon transforms 2D image which consists of class 1,2,3 and 4 to 3D
        image with 4 channel. Each channel contains either 255 or 0. (Similar to reverse of
        one-hot-encoding operation)
        :param img: predicted MRI mask
        :return: 3D image with number of channel equals to number of classes
        """
        four_channels = np.repeat(img[np.newaxis, :, :], self.num_classes, axis=0)
        for i in range(self.num_classes):
            four_channels[i][four_channels[i] == i] = 255
            four_channels[i][four_channels[i] != 255] = 0
        return four_channels.astype(np.float64)


# test code for dataset.py
if __name__ == "__main__":

    img = np.load('seg_masks/0.npy')
    mask = np.load('ground_truths/0.npy')

    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.show()
    keys = np.arange(len(os.listdir('seg_masks')) / 2, dtype=np.int16)
    data = ACDCseg(keys)
    img, mask = data[0]


