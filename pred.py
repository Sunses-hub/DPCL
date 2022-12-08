import argparse
from model import UNET2D
from dataset import ACDCseg
import torch
import os
import numpy as np
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from loss import SegmentationMetric
from dataset import add_blobs

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", default="daesaved_models/bestFold1dice919654.pth")
parser.add_argument("--img_dir", default="dae_data_old/seg_masks")
parser.add_argument("--label_dir", default="dae_data_old/ground_truths")
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--num_samples", type=int, default=4)


def inv_processor(img):
    """
    Input is assumed as torch tensor.
    """
    img = torch.tensor(img)
    img = softmax(img, dim=len(img.shape)-3)
    img = torch.argmax(img, dim=len(img.shape)-3)

    return img.numpy()


if __name__ == '__main__':
    parser = parser.parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    model = UNET2D(in_channels=parser.num_classes, out_channels=parser.num_classes).to(DEVICE)
    model.train(False)
    dict = torch.load(parser.model_dir,
                      map_location=lambda storage, loc: storage)
    save_model = dict["net"]
    model_dict = model.state_dict()
    # we only need to load the parameters of the encoder
    state_dict = {k: v for k, v in save_model.items()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


    # Find data size
    file_list = os.listdir(parser.img_dir)
    num_files = len(file_list)
    data_keys = np.array([int(file[:-4]) for file in file_list], dtype=np.int16)
    data = ACDCseg(keys=data_keys, img_dir=parser.img_dir, label_dir=parser.label_dir)

    samples = np.random.choice(range(len(data_keys)), size=1)

    metric_val = SegmentationMetric(parser.num_classes)

    # Display results
    with torch.no_grad():
        for i, idx in enumerate(samples):
            img, label = data[idx]
            # Input image
            input_img = inv_processor(img)
            # Prediction by the model
            pred = model(torch.unsqueeze(torch.tensor(img, dtype=torch.float), dim=0).to(DEVICE))
            # get dice
            metric_val.reset()
            dpred = softmax(pred, dim=1)
            metric_val.update(torch.tensor(label).long().squeeze(dim=1), dpred)
            _, _, dice = metric_val.get()

            pred_img = inv_processor(pred[0].cpu().detach().numpy())

            # display images
            # subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(1, 3)

            axarr[0].imshow(add_blobs(input_img))
            axarr[1].imshow(pred_img)
            axarr[2].imshow(label)
            plt.title(f"Pre-Denoiser vs. Post-Denoiser vs. Ground Truths (Dice = {np.around(dice, 3)})")
            plt.show()





