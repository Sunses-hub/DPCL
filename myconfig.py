
import argparse
import os

parser = argparse.ArgumentParser()

# Data
parser.add_argument("--patch_size", nargs='+', type=int)
parser.add_argument("--img_dir", type=str, default="seg_masks")
parser.add_argument("--label_dir", type=str, default="ground_truths")
parser.add_argument("--augmentation", default=False)
parser.add_argument("--train_ratio", type=float, default=0.80)
parser.add_argument("--class_num", type=int, default=4)
parser.add_argument("--M", type=int, default=100)

# Model
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--eta", type=float, default=1e-4) #1e-4
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_classes", type=int, default=4)

# DAE
parser.add_argument("--save_dir", default="dae_data_old")
parser.add_argument("--threshold", type=float, default=0.99)
parser.add_argument("--train_dae", default=False)
parser.add_argument("--model_dir", default="saved_models/bestFold1.pth")
parser.add_argument("--img_dae_dir", default="seg_masks")
parser.add_argument("--label_dae_dir", default="ground_truths")
# Training
parser.add_argument("--fold", type=int, default=5)

def get_config():
    config = parser.parse_args()
    config.patch_size = tuple(config.patch_size)
    return config