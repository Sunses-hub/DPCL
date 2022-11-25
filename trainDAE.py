# Necessary packages
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model import UNET2D
from dataset import SCD

def train_one_epoch(data_loader, device, model, optimizer, loss_fn, verbose=10):
    running_loss = 0
    loss_hist = []

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
            print(f"batch: {i+1}/{len(data_loader)} loss: {running_loss / verbose}")
            loss_hist.append(running_loss)
            running_loss = 0

    return loss_hist