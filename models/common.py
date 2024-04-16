import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        size: int = 4,
        padding: int = 1,
        stride: int = 2,
        apply_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        self.apply_batchnorm = apply_batchnorm

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=filters)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.apply_batchnorm:
            x = self.bn(x)

        return self.relu(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        size: int = 4,
        padding: int = 1,
        stride: int = 2,
        apply_dropout: bool = True,
    ) -> None:
        super().__init__()

        self.apply_dropout = apply_dropout

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=filters)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)

        if self.apply_dropout:
            x = self.dropout(x)

        return self.relu(x)
