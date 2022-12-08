# Necessary packages

import torch
from model import UNET2D
from torch.nn import CrossEntropyLoss
from torch import optim
from dataset import ACDCseg
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from sklearn.model_selection import KFold
from myconfig import get_config
from torch.nn.functional import softmax
from loss import SegmentationMetric
import os
from dataset import Blobs


def train_one_epoch(data_loader, device, model, optimizer, loss_fn, parser, verbose=10):

    """
    Train the model for one epoch over the given data.
    :param data_loader: data which is an instance of DataLoader class in PyTorch
    :param device: GPU or CPU
    :param model: 2D UNET model which is pretrained for segmentation
    :param optimizer: optimization algorithm
    :param loss_fn: loss function
    :param parser: other configurations given in my_config.py
    :param verbose: frequency of reporting results
    :return: cross-entropy loss and dice
    """

    running_loss = 0
    mean_loss = 0
    epoch_size = len(data_loader)

    # load dice evaluater
    metric_val = SegmentationMetric(parser.num_classes)
    metric_val.reset()
    # main training loop
    for i, batch in enumerate(data_loader):

        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs.float())
        # calculate dice score
        output = softmax(outputs, dim=1)
        metric_val.update(labels.long().squeeze(dim=1), output)
        # calculate cross-entropy loss or given any loss function
        loss = loss_fn(outputs, labels)
        loss.backward() # back-propagation
        # update weights
        optimizer.step()
        # add the loss to total loss
        running_loss += loss.item()
        mean_loss += loss.item()
        # report results if true
        if (i + 1) % verbose == 0:
            # Display loss
            mean_loss /= (verbose * inputs.shape[0])
            print(f"batch: {i+1}/{epoch_size} {loss_fn._get_name()}: {mean_loss}")
        elif (i+1) == epoch_size:
            mean_loss /= (epoch_size * parser.batch_size + inputs.shape[0]) # not sure
            print(f"batch {epoch_size}/{epoch_size} {loss_fn._get_name()}loss: {mean_loss}")
        mean_loss = 0
    # get segmentation metrics
    pixAcc, mIoU, Dice = metric_val.get()

    return running_loss, Dice

def run(config, train_keys, test_keys, fold_idx):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize model, loss, optimizer
    model = UNET2D(in_channels=config.num_classes, out_channels=config.num_classes).to(DEVICE)
    # initialize loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.eta)

    if config.augmentation:
        # Random Augmentations
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            Blobs()
        ])
    else:
        transform = None

    train_data = ACDCseg(keys=train_keys, img_dir=config.img_dir, label_dir=config.label_dir, transform=transform)
    test_data = ACDCseg(keys=test_keys, img_dir=config.img_dir, label_dir=config.label_dir)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers)
    train_loss = []
    val_loss = []
    dice_train_hist = []
    dice_val_hist = []

    dae = "dae" if config.train_dae else ''

    if not os.path.exists(os.path.join(os.getcwd(), dae + 'loss')):
        os.mkdir(dae + "loss")
    loss_dir = "loss"

    if not os.path.exists(os.path.join(os.getcwd(), dae + 'saved_models')):
        os.mkdir(dae + "saved_models")
    model_dir = dae + "saved_models"

    best_dice = 0

    print("*"*20 + f"FOLD {fold_idx}" + "*"*20)
    for epoch in range(config.num_epochs):
        print(f"EPOCHS : {epoch + 1}")

        model.train(True)
        # train for one epoch
        loss, dice_score = train_one_epoch(train_loader, DEVICE, model, optimizer, loss_fn, verbose=5, parser=config)
        epoch_mean_loss = loss / len(train_data)
        print(f"EPOCH {epoch + 1} MEAN {loss_fn._get_name().upper()}: {epoch_mean_loss}")
        print(f"EPOCH {epoch + 1} MEAN DICE SCORE: {dice_score}")
        train_loss.append(epoch_mean_loss)
        dice_train_hist.append(dice_score)
        if epoch >= 0:
            model.train(False)
            batch_val_loss = 0
            metric_val = SegmentationMetric(config.num_classes)
            metric_val.reset()
            # calculate validation performance and report
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    output = model(inputs.float())
                    batch_val_loss += loss_fn(output, labels).item()
                    # calculate dice score
                    output = softmax(output, dim=1)
                    metric_val.update(labels.long().squeeze(dim=1), output)

                pixAcc, mIoU, dice = metric_val.get()
                epoch_valm_loss = batch_val_loss / len(test_loader)
                val_loss.append(epoch_valm_loss)
                dice_val_hist.append(dice)
                print(f"Validation {loss_fn._get_name()} for Epoch {epoch + 1}:", epoch_valm_loss)
                print(f"Validation Dice for Epoch {epoch + 1}:", dice)
                if best_dice < dice:
                    best_dice = dice
                    # save model
                    save_dict = {"net": model.state_dict()}
                    for file in os.listdir(model_dir):
                        if f"bestFold{fold_idx}" in file:
                            os.remove(os.path.join(model_dir, file))
                    torch.save(save_dict, os.path.join(model_dir, f'bestFold{fold_idx}dice{str(best_dice)[2:8]}.pth'))

    # save the losses for different folds and train/val sets
    np.save(os.path.join(loss_dir, f"TrainLossFold{fold_idx}"), np.asarray(train_loss))
    np.save(os.path.join(loss_dir, f"ValLossFold{fold_idx}"), np.asarray(val_loss))
    # return the cross entropy loss and dice scores as a dictionary
    return {"train_loss" : np.array(train_loss),
            "val_loss" : np.array(val_loss),
            "dice_train" : np.array(dice_train_hist),
            "dice_val" : np.array(dice_val_hist)}

# run this code for testing whether training loop for denoiser works
if __name__ == "__main__":
    # initialize config
    config = get_config()

    num_files = len(os.listdir("seg_masks"))
    data_size = num_files / 2 if "0.png" in os.listdir("seg_masks") else num_files
    data_idxs = np.arange(data_size)

    train_mean_loss = np.zeros(config.num_epochs)
    val_mean_loss = np.zeros(config.num_epochs)
    train_mean_dice = np.zeros(config.num_epochs)
    val_mean_dice = np.zeros(config.num_epochs)

    # cross-validation
    kf = KFold(n_splits=config.fold, shuffle=True)
    for fold_idx, (train_keys, val_keys) in enumerate(kf.split(data_idxs)):

        hist_dict = run(config, train_keys, val_keys, fold_idx+1)
        tfold_loss, vfold_loss, tdice, valdice = hist_dict["train_loss"], hist_dict["val_loss"], hist_dict["dice_train"], hist_dict["dice_val"]

        train_mean_loss += tfold_loss / config.fold
        val_mean_loss += vfold_loss / config.fold
        train_mean_dice += tdice / config.fold
        val_mean_dice += valdice / config.fold

    # report the results
    print(f"{config.fold} FOLD TRAIN MEAN LOSS: {train_mean_loss}")
    print(f"{config.fold} FOLD VAL MEAN LOSS: {val_mean_loss}")
    print(f"{config.fold} FOLD TRAIN MEAN DICE: {train_mean_dice}")
    print(f"{config.fold} FOLD VAL MEAN DICE: {val_mean_dice}")

    np.save(f"{config.fold}foldTrainMeanLoss", train_mean_loss)
    np.save(f"{config.fold}foldValMeanLoss", val_mean_loss)
    np.save(f"{config.fold}foldTrainMeanDice", train_mean_dice)
    np.save(f"{config.fold}foldValMeanDice", val_mean_dice)
