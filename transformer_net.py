# style_transfer/models/transform_net.py
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            ConvLayer(channels, channels, 3, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upsample)
        return self.layer(x)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleConvLayer(128, 64, 3, 1, upsample=2),
            UpsampleConvLayer(64, 32, 3, 1, upsample=2),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, 1)
        )

    def forward(self, x):
        return self.model(x)
