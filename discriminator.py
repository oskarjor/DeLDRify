# copied from https://github.com/Lornatang/ESRGAN-PyTorch

import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

class DiscriminatorForVGG(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            depth: int = 4
    ) -> None:
        self.out_channels = out_channels
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(channels, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(int(2 * channels), int(2 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(int(4 * channels), int(4 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(int(8 * channels), int(8 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(int(8 * channels), int(8 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(8 * channels) * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, out_channels)
        )
        

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
    

class DiscriminatorForVGG2(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            depth: int = 4
    ) -> None:
        assert depth >= 0 and depth <= 4, "depth must be in [0, 4]"
        self.out_channels = out_channels
        super(DiscriminatorForVGG2, self).__init__()
        self.first_conv = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
        )
        self.features = nn.ModuleList()
        for _ in range(depth):
            self.features.append(self.block(channels))
            channels *= 2
        self.final_conv = nn.Sequential(
            # state size. (512) x 4 x 4
            nn.Conv2d(int(channels), int(channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(channels)),
            nn.LeakyReLU(0.2, True), 
            # state size. (512) x 4 x 4
            nn.Conv2d(int(channels), int(channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(channels)),
            nn.LeakyReLU(0.2, True), 
            # state size. (512) x 4 x 4
            nn.Conv2d(int(channels), int(channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(channels)),
            nn.LeakyReLU(0.2, True), 
        )

        space_dim = 2 ** (5 - depth)

        self.classifier = nn.Sequential(
            nn.Linear(int(channels) * space_dim * space_dim, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, out_channels)
        )

    def block(self, channels) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
        )
        

    def forward(self, x: Tensor) -> Tensor:
        out = self.first_conv(x)
        for feat in self.features:
            out = feat(out)
        out = self.final_conv(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out