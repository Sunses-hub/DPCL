
import argparse
import os

parser = argparse.ArgumentParser()

# Data
parser.add_argument("--patch_size", nargs='+', type=int)
parser.add_argument("--img_dir", type=str, default="seg_masks")
parser.add_argument("--label_dir", type=str, default="ground_truths")
parser.add_argument("--augmentation", default=False)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--class_num", type=int, default=4)

# Model
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--eta", type=float, default=1e-4)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)


def get_config():
    config = parser.parse_args()
    config.patch_size = tuple(config.patch_size)
    return config