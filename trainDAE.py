# Necessary packages

import torch
from trainUNET2D import train_one_epoch
from model import DAE
from loss import DiceLoss
from torch import optim
from dataset import SCD
from torch.utils.data import DataLoader
import numpy as np

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

    train_data = SCD(train=True, noisy=True, root_dir='Data')
    val_data = SCD(train=False, noisy=True, root_dir='Data')

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)

    train_loss = []
    val_loss = []

    for epoch in range(NUM_EPOCHS):
        print(f"EPOCHS : {epoch+1}")

        model.train(True)
        loss = train_one_epoch(train_dataloader, DEVICE, model, optimizer, loss_fn, verbose=5)
        epoch_mean_loss = np.sum(loss) / len(train_dataloader)
        print(f"EPOCH {epoch+1} MEAN LOSS: {epoch_mean_loss}")
        train_loss.append(epoch_mean_loss)

        model.eval()
        batch_val_loss = 0
        for i, batch in enumerate(val_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            output = model(inputs.float())
            batch_val_loss += loss_fn(output, labels).item()
        epoch_valm_loss = batch_val_loss / len(val_dataloader)
        val_loss.append(epoch_valm_loss)
        print(f"Validation DICE Loss for Epoch {epoch+1}:", epoch_valm_loss)

    np.save("DAE_train_loss", np.asarray(train_loss))
    np.save("DAE_val_loss", np.asarray(val_loss))