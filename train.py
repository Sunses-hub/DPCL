# Necessary packages
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import UNET2D
import SCD


def train_one_epoch(data_loader, device, model, optimizer, loss_fn, verbose=10):
    running_loss = 0
    loss = []

    for i, batch in enumerate(data_loader):

        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)

        loss = loss_fn(output, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % verbose == 0:
            # Display loss
            print(f"batch: {i}/{np.around(len(data_loader) / verbose)} loss: {running_loss / verbose}")
            loss.append(running_loss)
            running_loss = 0

    return loss

if __name__ == "__main__":

    # Initializations
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    NUM_WORKERS = 2

    model = UNET2D(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_data = SCD(train=True, root_dir='Data')
    val_data = SCD(train=False, root_dir='Data')

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)

    train_loss = []
    val_loss = []

    for epoch in range(NUM_EPOCHS):
        print(f"EPOCHS : {epoch}")

        model.train(True)
        loss = train_one_epoch(train_dataloader, DEVICE, model, optimizer, loss_fn, verbose=5)
        train_loss.append(loss)

        model.eval()
        mean_val = 0
        for i, batch in enumerate(val_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            output = model(inputs)
            loss = loss_fn(output, labels).item()
            val_loss.append(loss)
            mean_val += loss/len(val_dataloader)
        print(f"Validation Loss for Epoch {epoch+1}:", mean_val)

    np.save(np.asarray(train_loss))
    np.save(np.asarray(val_loss))

