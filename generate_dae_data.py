


import argparse
from model import UNET2D
from dataset import ACDCseg
import torch
import numpy as np
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from loss import SegmentationMetric
import os

# configurations for generating a dataset for denoiser (PDCL)
parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", default="saved_models/best.pth")
parser.add_argument("--img_dir", default="seg_masks")
parser.add_argument("--label_dir", default="ground_truths")
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

parser.add_argument("--save_dir", default="dae_data_old")
parser.add_argument("--threshold", type=float, default=0.9)

def generate_dae_data(parser):

    """
    Loads the pre-trained model and calculating predictions it generates a dataset
    for denoiser. In the dataset, samples which has dice score less than the thre-
    shold are kept. Others are discarded because they are good enough.
    :param parser: configurations
    :return: returns the data
    """

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # initialize the model to be loaded
    model = UNET2D(in_channels=parser.num_classes, out_channels=parser.num_classes).to(DEVICE)
    model.train(False)
    dict = torch.load(parser.model_dir,
                      map_location=lambda storage, loc: storage)
    # load state dictionary of the model
    save_model = dict["net"]
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # Find data and load its part which corresponds to data_keys (indices)
    num_files = len(os.listdir("seg_masks"))
    data_size = num_files
    data_keys = np.arange(data_size, dtype=np.int16)
    data = ACDCseg(keys=data_keys, img_dir=parser.img_dir, label_dir=parser.label_dir)
    dataloader = DataLoader(data, batch_size=1)

    # create directories for data
    if not os.path.exists(parser.save_dir):
        os.mkdir(parser.save_dir)
    # directory for predictions of segmentation model
    save_data = os.path.join(parser.save_dir, "seg_masks")
    if not os.path.exists(save_data):
        os.mkdir(save_data)
    # directory for ground truths of the corresponding samples
    save_labels = os.path.join(parser.save_dir, "ground_truths")
    if not os.path.exists(save_labels):
        os.mkdir(save_labels)

    data_keys = [] # data keys to be used for the training of the denoiser
    metric_val = SegmentationMetric(parser.num_classes) # metric for calculating dice score
    model.train(False)
    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            metric_val.reset()
            inputs, labels = sample
            inputs, labels = inputs.float().to(DEVICE), labels.long().to(DEVICE)
            output = model(inputs) # forward pass
            # For dice score
            output = softmax(output, dim=1) # calculate probs.
            metric_val.update(labels.long().squeeze(dim=1), output) # calculate mean dice score
            pixAcc, mIoU, dice = metric_val.get() # get dice score
            # save the sample if its dice score is less than threshold, otherwise discard it
            if dice < parser.threshold:
                np.save(file=os.path.join(save_data, f"{idx}"), arr=torch.argmax(output[0], dim=0).cpu().numpy()+1)
                np.save(file=os.path.join(save_labels, f"{idx}"), arr=labels[0].cpu().numpy()+1)
                print(f"Sample {idx} was saved.")
                print("Dice score for that sample:", dice)
                print("-" * 30)
                data_keys.append(idx) # add the relevant key for later use

    print("All files were saved.")
    return np.array(data_keys)

# Test for generating data of DAE
if __name__ == "__main__":
    parser = parser.parse_args()
    data_keys = generate_dae_data(parser)
    print("Number of data to be used for DAE is", len(data_keys))
