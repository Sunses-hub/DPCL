# Necessary packages

import torch
from trainUNET2D import train_one_epoch
from model import UNET2D
from torch.nn import CrossEntropyLoss
from torch import optim
from dataset import ACDCseg
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from myconfig import get_config
import os



if __name__ == "__main__":
    # initialize config
    config = get_config()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers
    num_file = len(os.listdir(config.img_dir))
    DATA_SIZE = num_file / 2 if os.path.join(config.img_dir, "0.png") in os.listdir(config.img_dir) else num_file

    model = UNET2D(in_channels=config.class_num, out_channels=config.class_num)
    loss_fn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.eta)

    if config.augmentation:
        # Augmentation
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            ], p=0.8),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
            ToTensorV2(),
        ])
    else:
        transform = None

    train_keys, test_keys = train_test_split(np.arange(DATA_SIZE), train_size=config.train_ratio, shuffle=True)

    train_data = ACDCseg(keys=train_keys, img_dir=config.img_dir, label_dir=config.label_dir, transform=transform)
    test_data = ACDCseg(keys=test_keys, img_dir=config.img_dir, label_dir=config.label_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers)
    train_loss = []
    val_loss = []

    for epoch in range(config.num_epochs):
        print(f"EPOCHS : {epoch+1}")

        model.train(True)
        loss = train_one_epoch(train_loader, DEVICE, model, optimizer, loss_fn, verbose=5)
        epoch_mean_loss = loss / len(train_data)
        print(f"EPOCH {epoch+1} MEAN LOSS: {epoch_mean_loss}")
        train_loss.append(epoch_mean_loss)

        model.train(False)
        batch_val_loss = 0
        for i, batch in enumerate(test_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            output = model(inputs.float())
            batch_val_loss += loss_fn(output, labels).item()
        epoch_valm_loss = batch_val_loss / len(test_loader)
        val_loss.append(epoch_valm_loss)
        print(f"Validation DICE Loss for Epoch {epoch+1}:", epoch_valm_loss)

    np.save("DAE_train_loss", np.asarray(train_loss))
    np.save("DAE_val_loss", np.asarray(val_loss))