import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import ACDCseg
from torch.nn import CrossEntropyLoss
from loss import SegmentationMetric
from model import UNET2D
from torch.nn.functional import softmax
from pred import inv_processor
import matplotlib.pyplot as plt

# This code evaluates the final model's performance
parser = argparse.ArgumentParser()

# Models
parser.add_argument("--seg_model_dir", default="saved_models/bestFold1.pth")
parser.add_argument("--dae_model_dir", default="daesaved_models/bestFold1dice945652.pth")

# Data
parser.add_argument("--train_img_dir", default="seg_masks")
parser.add_argument("--train_label_dir", default="ground_truths")
parser.add_argument("--test_img_dir", default="seg_masks")
parser.add_argument("--test_label_dir", default="ground_truths")
parser.add_argument("--num_classes", type=int, default=4)

# Other parameters
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int,  default=2)
parser.add_argument("--verbose", type=int, default=10)

def load_model(model_dir, config):

    """
    Loads a PyTorch model from given model directory (model_dir) with given
    configurations (config).
    :param model_dir: directory of the model to be loaded
    :param config: configurations of the model (comes from my_config.py)
    :return:
    """
    # initialize model
    model = UNET2D(in_channels=config.num_classes, out_channels=config.num_classes)
    model.eval() # evaluation mode on
    dict = torch.load(model_dir,
                      map_location=lambda storage, loc: storage)
    save_model = dict["net"]
    model_dict = model.state_dict()
    # load model
    state_dict = {k: v for k, v in save_model.items()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print("Model is loaded from", model_dir)
    return model

def image_processor(pred_batch, DEVICE):
    """
    Does the same job as the image_processor of ACDCseg dataset class with addition of softmax.
    :param pred_batch: batch of predicted segmentation masks, it is expected as (batch size, num. of class, height, width)
    :param DEVICE: device to train the algorithm (use cuda GPU for faster convergence)
    :return:
    """
    pred_prob = softmax(pred_batch, dim=1)
    labels = torch.argmax(pred_prob, dim=1)
    out = torch.zeros(pred_prob.shape).to(DEVICE).scatter(1, labels.unsqueeze(1), 255.0)
    return out


if __name__ == "__main__":

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = parser.parse_args()

    noisy_dices = []
    final_dices = []

    test_size = len(os.listdir(config.test_img_dir)) # data size
    test_keys = np.arange(test_size)

    # create dataset
    test_data = ACDCseg(keys=test_keys, img_dir=config.test_img_dir, label_dir=config.test_label_dir)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)

    # Loss functions
    cross_entropy = CrossEntropyLoss()
    metric_val = SegmentationMetric(config.num_classes)
    noisy_metric_val = SegmentationMetric(config.num_classes)
    metric_val.reset()

    # Load models
    seg_model = load_model(config.seg_model_dir, config).to(DEVICE)
    dae_model = load_model(config.dae_model_dir, config).to(DEVICE)

    # Test Data evaluation
    metric_val.reset()

    total_ce_loss = 0 # total cross entropy loss
    print("-" * 20, "EVALUATION", "-" * 20)
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, labels, _ = batch
            images, labels = images.float().to(DEVICE), labels.float().to(DEVICE)
            # Forward pass pipeline
            noisy_metric_val.update(labels.long().squeeze(dim=1), images)
            _, _, noisy_dice = noisy_metric_val.get()
            noisy_dices.append(noisy_dice)
            noisy_metric_val.reset()
            final_pred = dae_model(images) # predictions of segmentation model

            # For dice score
            probs = softmax(final_pred, dim=1)
            noisy_metric_val.update(labels.long().squeeze(dim=1), probs)
            _, _, final_dice = noisy_metric_val.get()
            final_dices.append(final_dice)
            noisy_metric_val.reset()
            metric_val.update(labels.long().squeeze(dim=1), probs)

            _, _, dice = metric_val.get()
            # For CrossEntropyLoss
            ce_loss = cross_entropy(final_pred, labels.long())
            total_ce_loss += ce_loss.item()
            # Report losses
            if (idx + 1) % config.verbose == 0:
                mean_loss = total_ce_loss / ((idx + 1) * images.shape[0])
                print(f"batch: {idx + 1}/{len(test_loader)} Cross Entropy Mean Loss: {mean_loss}")
                print(f"batch: {idx + 1}/{len(test_loader)} Mean Dice Score: {dice}")
                print("-" * 65)
            elif (idx + 1) == len(test_loader):
                mean_loss = total_ce_loss / len(test_data)
                print(f"batch: {idx + 1}/{len(test_loader)} Cross Entropy Mean Loss: {mean_loss}")
                print(f"batch: {idx + 1}/{len(test_loader)} Mean Dice Score: {dice}")
                print("-" * 65)

    _, _, dice = metric_val.get()
    # Final Report
    noisy_dices = np.array(noisy_dices)
    final_dices = np.array(final_dices)
    diff = final_dices - noisy_dices
    # Choose the top and the worst performances
    best_performers = np.flip(np.argsort(diff)[-20:])
    worst_performers = np.argsort(diff)[:20]

    # Plot and save best and worst performers
    for idx in best_performers:
        img, label, mri = test_data[idx]

        nz = np.nonzero(mri)  # Indices of all nonzero elements
        mri = mri[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f'M=2, DICE before={noisy_dices[idx]}, DICE after={final_dices[idx]}, change={diff[idx]}')
        for i in range(3):
            axs[i].imshow(mri, cmap='gray')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        # process image to put it into displayable form
        processed_img = inv_processor(img)
        processed_img = np.ma.masked_where(processed_img == 0, processed_img)
        processed_img = processed_img[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        # plot pre-denoiser prediction
        axs[0].imshow(processed_img, alpha=0.5)
        axs[0].set_title("Pre-denoiser",fontdict={"fontsize":7})
        # get predictions from denoiser model
        final_pred = dae_model(torch.tensor(img[None], dtype=torch.float32).to(DEVICE))
        # get the image into displayable form
        processed_final = inv_processor(final_pred[0].cpu().detach().numpy())
        processed_final = np.ma.masked_where(processed_final == 0, processed_final)
        processed_final = processed_final[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        # plot post-denoiser prediction
        axs[1].imshow(processed_final, alpha=0.5)
        axs[1].set_title("Post-denoiser",fontdict={"fontsize":7})
        # get ground truths
        label = np.ma.masked_where(label == 0, label)
        label = label[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        # plot ground truths
        axs[2].imshow(label, alpha=0.5)
        axs[2].set_title("Ground-truth",fontdict={"fontsize":7})
        # save the figure
        plt.savefig(f"figures/best-80/{idx}",bbox_inches='tight', dpi=200)

    for idx in worst_performers:
        img, label, mri = test_data[idx]

        nz = np.nonzero(mri)  # Indices of all nonzero elements
        mri = mri[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f'M=2, DICE before={noisy_dices[idx]}, DICE after={final_dices[idx]}, change={diff[idx]}')
        # put mri images to different subfigures
        for i in range(3):
            axs[i].imshow(mri, cmap='gray')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        processed_img = inv_processor(img)
        processed_img = np.ma.masked_where(processed_img == 0, processed_img)
        processed_img = processed_img[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        # plot pre-denoiser (segmentation model) prediction
        axs[0].imshow(processed_img, alpha=0.5)
        axs[0].set_title("Pre-denoiser",fontdict={"fontsize":7})
        # get final predictions of post-denoiser
        final_pred = dae_model(torch.tensor(img[None], dtype=torch.float32).to(DEVICE))
        # process image to make it displayable form
        processed_final = inv_processor(final_pred[0].cpu().detach().numpy())
        processed_final = np.ma.masked_where(processed_final == 0, processed_final)
        processed_final = processed_final[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        # plot post-denoiser prediction
        axs[1].imshow(processed_final, alpha=0.5)
        axs[1].set_title("Post-denoiser",fontdict={"fontsize":7})
        # process ground truths for plotting
        label = np.ma.masked_where(label == 0, label)
        label = label[nz[0].min():nz[0].max() + 1, nz[1].min():nz[1].max() + 1]
        # plot ground truths
        axs[2].imshow(label, alpha=0.5)
        axs[2].set_title("Ground-truth",fontdict={"fontsize":7})
        # save the figure
        plt.savefig(f"figures/worst-80/{idx}",bbox_inches='tight',dpi=200)


    print("EVALUATION IS COMPLETED")
    print("MEAN DICE SCORE:", dice)
    print("BEST IMRPOVEMENTS:", best_performers)
    print("WORST FAILERS:", worst_performers)
    print("-" * 63)


