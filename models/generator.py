import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, stride=2, padding=1, use_leaky_relu=True, transposed=False, apply_batch_norm=True):
        super().__init__()
        if transposed:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels) if apply_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU()

    def forward(self, x, apply_dropout=False):
        x = self.conv(x)
        x = self.batch_norm(x)
        if apply_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self, out_dims):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        in_channels = 3
        for i, out_dim in enumerate(out_dims):
            apply_batch_norm = False if i == 0 else True
            self.conv_blocks.append(ConvBlock(in_channels, out_dim, apply_batch_norm=apply_batch_norm))
            in_channels = out_dim

    def forward(self, x):
        skips = []
        for block in self.conv_blocks:
            x = block(x)
            skips.append(x)
        return x, skips

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, transposed=False)

    def forward(self, x):
        return self.conv_block(x)

class Decoder(nn.Module):
    def __init__(self, skip_dims, in_channels=512):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.in_block = ConvBlock(in_channels, in_channels, use_leaky_relu=False, transposed=True)
        self.out_block = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        for skip_dim in reversed(skip_dims[1:]):
            self.conv_blocks.append(
                ConvBlock(in_channels + skip_dim, skip_dim, use_leaky_relu=False, transposed=True)
            )
            in_channels = skip_dim
        self.tanh = nn.Tanh()

    def forward(self, x, skips):
        x = self.in_block(x, apply_dropout=True)
        for i, (block, skip) in enumerate(zip(self.conv_blocks, reversed(skips[1:]))):
            x = torch.cat([x, skip], dim=1)
            apply_dropout = i in [0, 1, 2]
            x = block(x, apply_dropout)
        x = self.out_block(x)
        return self.tanh(x)

class Generator(nn.Module):
    def __init__(self, encoder_dims, bottleneck_dim):
        super().__init__()
        self.encoder = Encoder(encoder_dims)
        self.bottleneck = Bottleneck(bottleneck_dim, bottleneck_dim)
        self.decoder = Decoder(encoder_dims)

    def forward(self, x):
        encoded, skips = self.encoder(x)
        bottleneck_out = self.bottleneck(encoded)
        return self.decoder(bottleneck_out, skips)
