# Necessary packages

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


# Post DAE Implementation
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encode(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)

# Another denoiser algorithm but it is not used
class PostDAE(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(PostDAE, self).__init__()
        # Encoder
        self.L1 = DownConv(in_channels, 16)
        self.L2 = DownConv(16, 32)
        self.L3 = DownConv(32, 32)
        self.L4 = DownConv(32, 32)
        self.L5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=2),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        # FC
        self.L6 = nn.Sequential(nn.Linear(2048, 4096),
                                nn.BatchNorm1d(4096),
                                nn.ReLU(inplace=True))
        # Decoder
        self.L8 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )#UpConv(16, 16)

        self.L9 = UpConv(16, 16)
        self.L10 = UpConv(16, 16)
        self.L11 = UpConv(16, 16)
        self.L12 = nn.Sequential(nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(16, out_channels, kernel_size=4, stride=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),

        )

        self.network = nn.ModuleList([self.L1, self.L2, self.L3, self.L4, self.L5,
                                      self.L6,
                                      self.L8, self.L9, self.L10, self.L11, self.L12])

    def forward(self, x):
        # Encoder
        h = x
        for i in range(5):
            layer = self.network[i]
            h = layer(h)
        # Fully connected layer
        h = h.view(h.size(0), -1)
        h = self.L6(h)
        tmp = h.size(1)
        height = int(tmp / 16)
        h = h.view(h.size(0), 16, int(height ** 0.5), int(height ** 0.5))
        # Decoder
        for i in range(5):
            layer = self.network[6+i]
            h = layer(h)

        return h




# UNET2D implementation

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.encode(x)


class UNET2D(nn.Module):

    """
    Here, we implemented the original 2D UNET architecture in the original paper: https://arxiv.org/abs/1505.04597
    Different than the paper, we used 2D Batch Normalizaton and adjust the network for images size of 352x352.
    """

    def __init__(self, in_channels=4, out_channels=4):
        super(UNET2D, self).__init__()

        self.channels = [64, 128, 256, 512]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        old_feature = in_channels
        # Initialize the encoder
        for num_feature in self.channels:
            self.encoder.append(DoubleConv(old_feature, num_feature))
            old_feature = num_feature

        self.bottleneck = DoubleConv(512, 1024)

        # Initialize the decoder
        old_feature = self.channels[-1] * 2
        for num_feature in reversed(self.channels):
            self.decoder.append(nn.ConvTranspose2d(old_feature, num_feature, 2, 2))
            self.decoder.append(DoubleConv(old_feature, num_feature))
            old_feature = num_feature

        self.head = nn.Conv2d(self.channels[0], out_channels, 1)

    def forward(self, x):

        layer_outputs = []
        # Going through the encoder
        out = x
        for layer in self.encoder:
            out = layer(out)
            layer_outputs.append(out)
            out = self.max_pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Going through the decoder
        for idx, layer in enumerate(self.decoder):
            out = layer(out)
            if idx % 2 == 0:
                connection = F.resize(layer_outputs[-(1 + idx // 2)], out.shape[2:])
                out = torch.cat((connection, out), dim=1)

        out = self.head(out)
        return out


# test
if __name__ == "__main__":
    print("UNET2D Test")
    tmp = torch.rand(8,3,256,256)
    model = UNET2D(in_channels=3, out_channels=1)
    output = model.forward(tmp)
    print(f"Input size: {tmp.shape}")
    print(f"Output size: {output.shape}")