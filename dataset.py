# Necessary packages
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt

class ACDCseg(Dataset):

    def __init__(self, keys, img_dir='seg_masks', label_dir='ground_truths', transform=None):

        self.img_files = [os.path.join(img_dir, str(file_name) + '.npy') for file_name in keys]
        self.mask_files = [os.path.join(label_dir, str(file_name) + '.npy') for file_name in keys]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        I will leave the class indices for mask because of the CrossEntropy loss function
        but doing that may be wrong, so check that later.
        """
        img, mask = np.load(self.img_files[index]), np.load(self.mask_files[index])
        img = ACDCseg.img_processor(img)
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']
        return img, mask

    @staticmethod
    def img_processor(img):
        num_classes = len(np.unique(img))
        new_img = np.zeros((num_classes, img.shape[0], img.shape[1]))
        for row_idx in range(img.shape[0]):
            for col_idx in range(img.shape[1]):
                channel_num = img[row_idx, col_idx]
                new_img[channel_num-1, row_idx, col_idx] = 255
        return new_img



class SCD(Dataset):
    """Sunny Brook Cardiac Imaging Dataset"""

    def __init__(self, train=True, root_dir='Data', transform=None, noisy=False):

        self.root_dir = os.path.join(root_dir, 'Train') if train else os.path.join(root_dir, 'Test')
        self.img_dirs = []
        self.label_dirs = []
        self.transform = transform
        self.noisy = noisy

        for root, dirs, files in os.walk(self.root_dir):
            dir_name = os.path.basename(root)
            if len(files) != 0 and ('Sunnybrook' in dir_name):
                for file in sorted(files):
                    if 'Images' in root:
                        self.img_dirs.append(os.path.join(dir_name, file))
                    else:
                        self.label_dirs.append(os.path.join(dir_name, file))

        if noisy:
            self.noise = np.random.normal(loc=0, scale=1,size=(len(self.img_dirs), 256, 256))

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()


        mask = io.imread(os.path.join(self.root_dir, 'Labels', self.label_dirs[index]), as_gray=True).astype('float32')

        if self.noisy:
            image = mask + self.noise[index]
        else:
            image = io.imread(os.path.join(self.root_dir, 'Images', self.img_dirs[index])).astype('float32')
            image = np.transpose(image, axes=(2, 0, 1))
            mask = np.expand_dims(mask, axis=0)

            if self.transform:
                image = self.transform(image)

        return image, mask


# test
if __name__ == "__main__":
    print("Test for Sunnybrook dataset")
    path = os.path.join("Data", "Train", "Images", "Sunnybrook_Part1")
    annotations = os.listdir(path)
    img_file = os.path.join(path, annotations[10])
    img = io.imread(img_file, as_gray=True)

    plt.imshow(img)
    plt.show()

    print("Test for Noisy Sunnybrook dataset")
    data = SCD(train=True, noisy=True, root_dir='Data')
    print("Length:", len(data))
    noisy_mask, ground_truth = data[0]
    print("Noisy Mask shape:", noisy_mask.shape)
    print("Ground Truth shape:", ground_truth.shape)
    noisy_mask = np.asarray(noisy_mask[0])
    plt.imshow(noisy_mask)
    plt.show()
    ground_truth = np.asarray(ground_truth[0])
    plt.imshow(ground_truth)
    plt.show()