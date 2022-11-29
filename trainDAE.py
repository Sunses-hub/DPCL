# Necessary packages

import torch
from trainUNET2D import train_one_epoch
from model import DAE
from loss import DiceLoss
from torch import optim
from dataset import ACDCseg
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from torch.utils.data import random_split

if __name__ == "__main__":
    # Initializations
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    NUM_WORKERS = 2

    model = DAE(IMAGE_WIDTH * IMAGE_HEIGHT)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #train_data = SCD(train=True, noisy=True, root_dir='Data')
    #val_data = SCD(train=False, noisy=True, root_dir='Data')

    # Augmentation
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5)
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
        ], p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)
    ])

    data = ACDCseg(train=True, img_dir='seg_masks', label_dir='ground_truths', transform=transform)
    train_set, val_set = random_split(data, [0.8, 0.2])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2)

    train_loss = []
    val_loss = []

    for epoch in range(NUM_EPOCHS):
        print(f"EPOCHS : {epoch+1}")

        model.train(True)
        loss = train_one_epoch(train_loader, DEVICE, model, optimizer, loss_fn, verbose=5)
        epoch_mean_loss = loss / len(train_loader)
        print(f"EPOCH {epoch+1} MEAN LOSS: {epoch_mean_loss}")
        train_loss.append(epoch_mean_loss)

        model.eval()
        batch_val_loss = 0
        for i, batch in enumerate(val_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            output = model(inputs.float())
            batch_val_loss += loss_fn(output, labels).item()
        epoch_valm_loss = batch_val_loss / len(val_loader)
        val_loss.append(epoch_valm_loss)
        print(f"Validation DICE Loss for Epoch {epoch+1}:", epoch_valm_loss)

    np.save("DAE_train_loss", np.asarray(train_loss))
    np.save("DAE_val_loss", np.asarray(val_loss))