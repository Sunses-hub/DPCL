# Necessary packages

import torch
import torch.nn as nn
import torchvision.transforms.functional as F



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

    def __init__(self, in_channels=3, out_channels=1):
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


class DAE(nn.Module):
    def __init__(self, num_features):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=num_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)

# test
if __name__ == "__main__":
    print("UNET2D Test")
    tmp = torch.rand(8,3,256,256)
    model = UNET2D(in_channels=3, out_channels=1)
    output = model.forward(tmp)
    print(f"Input size: {tmp.shape}")
    print(f"Output size: {output.shape}")

    print("DAE Test")
    denoiser_model = DAE(256*256)
    random_data = torch.rand(8, 1, 256, 256)
    flattened = torch.flatten(random_data, start_dim=2, end_dim=3)
    print("Input size:", flattened.shape)
    print("Output size:", denoiser_model(flattened).shape)