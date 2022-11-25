# Necessary packages
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt

class SCD(Dataset):
    """Sunny Brook Cardiac Imaging Dataset"""

    def __init__(self, train=True, root_dir='Data', transform=None):

        self.root_dir = os.path.join(root_dir, 'Train') if train else os.path.join(root_dir, 'Test')
        self.img_dirs = []
        self.label_dirs = []
        self.transform = transform

        for root, dirs, files in os.walk(self.root_dir):
            dir_name = os.path.basename(root)
            if len(files) != 0 and ('Sunnybrook' in dir_name):
                for file in sorted(files):
                    if 'Images' in root:
                        self.img_dirs.append(os.path.join(dir_name, file))
                    else:
                        self.label_dirs.append(os.path.join(dir_name, file))

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        image = io.imread(os.path.join(self.root_dir, 'Images', self.img_dirs[index])).astype('float32')
        mask = io.imread(os.path.join(self.root_dir, 'Labels', self.label_dirs[index]), as_gray=True).astype('float32')
        image = np.transpose(image, axes=(2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image = self.transform(image)

        return image, mask

# test
if __name__ == "__main__":
    path = os.path.join("Data", "Train", "Images", "SunnyBrook_Part1")
    annotations = os.listdir(path)
    img_file = os.path.join(path, annotations[10])
    img = io.imread(img_file, as_gray=True)

    plt.imshow(img)
    plt.show()