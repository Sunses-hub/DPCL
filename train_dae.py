import os
import numpy as np
import torch
from generate_dae_data import generate_dae_data
from myconfig import get_config
from sklearn.model_selection import ShuffleSplit
from train import run

# this is not a test code. run this code to train 2D UNET denoiser. 
if __name__ == "__main__":

    config = get_config()

    torch.manual_seed(10) # for reproducible results
    # if DAE data was not created, generate it. otherwise, skip generation step.
    if len(os.listdir(os.path.join(config.save_dir, config.img_dae_dir))) != 0:
        print("Data for DAE was already created. No need to load again.")
        data_keys = np.arange(len(os.listdir(os.path.join(config.save_dir, config.img_dae_dir))))
        print(f"{len(data_keys)} sample was found.")
    else:
        print("Data for DAE is being generated...")
        # initialize config
        data_keys = generate_dae_data(config)

    total_data = config.M * 19 + 2 # choose number of slices
    print(f"{config.M} patients are being used for training.")
    print(f"{total_data} slices are chosen.")

    train_mean_loss = np.zeros(config.num_epochs)
    val_mean_loss = np.zeros(config.num_epochs)
    train_mean_dice = np.zeros(config.num_epochs)
    val_mean_dice = np.zeros(config.num_epochs)

    # cross-validation loop
    ss = ShuffleSplit(n_splits=config.fold, train_size=total_data)
    for fold_idx, (train_keys, val_keys) in enumerate(ss.split(np.zeros(len(data_keys)), data_keys)):
        hist_dict = run(config, train_keys, val_keys, fold_idx + 1)
        tfold_loss, vfold_loss, tdice, valdice = hist_dict["train_loss"], hist_dict["val_loss"], hist_dict["dice_train"], hist_dict["dice_val"]
        # calculate mean performances
        train_mean_loss += tfold_loss / config.fold
        val_mean_loss += vfold_loss / config.fold
        train_mean_dice += tdice / config.fold
        val_mean_dice += valdice / config.fold

    print(f"{config.fold} FOLD TRAIN MEAN LOSS: {train_mean_loss}")
    print(f"{config.fold} FOLD VAL MEAN LOSS: {val_mean_loss}")
    print(f"{config.fold} FOLD TRAIN MEAN DICE: {train_mean_dice}")
    print(f"{config.fold} FOLD VAL MEAN DICE: {val_mean_dice}")


    if not os.path.exists("daeMeanLoss"):
        os.mkdir("daeMeanLoss")
    # save the results
    np.save(os.path.join("daeMeanLoss", f"{config.fold}foldTrainMeanLoss"), train_mean_loss)
    np.save(os.path.join("daeMeanLoss", f"{config.fold}foldValMeanLoss"), val_mean_loss)
    np.save(os.path.join("daeMeanLoss", f"{config.fold}foldTrainMeanDice"), train_mean_dice)
    np.save(os.path.join("daeMeanLoss", f"{config.fold}foldValMeanDice"), val_mean_dice)
