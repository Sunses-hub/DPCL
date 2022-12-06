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
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)

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
                                 nn.Conv2d(16, out_channels, kernel_size=4, stride=1)
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


class UNET2D(nn.Module):

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


class DAE(nn.Module):
    def __init__(self, num_features):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(in_features=num_features, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)

# test
if __name__ == "__main__":
    '''
    print("UNET2D Test")
    tmp = torch.rand(8,3,256,256)
    model = UNET2D(in_channels=3, out_channels=1)
    output = model.forward(tmp)
    print(f"Input size: {tmp.shape}")
    print(f"Output size: {output.shape}")

    print("DAE Test")
    denoiser_model = DAE(256*256)
    random_data = torch.rand(16, 256, 256)
    print("Input size:", random_data.shape)
    print("Output size:", denoiser_model(random_data).shape)
    
    '''

    print("POST-DAE TEST")
    tmp = torch.rand(8,4,352,352)
    model = PostDAE(in_channels=4, out_channels=4)
    output = model.forward(tmp)
    print(f"Input shape: {tmp.shape}")
    print(f"Output shape: {output.shape}")

