
import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

# test
if __name__ == "__main__":
    loss = DiceLoss()
    rand1 = torch.rand(1,256,256)
    rand2 = torch.rand(1,256,256)
    dice = loss(rand1, rand2)
    print("Dice score:", dice.item())