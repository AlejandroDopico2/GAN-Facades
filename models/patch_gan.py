import torch
import torch.nn as nn

from models.common import EncoderBlock

class Discriminator(nn.Module):
    def __init__(self ) -> None:
        super().__init__()

        filters = 32

        self.encoder = nn.ModuleList()
        for i in range(3):
            if i == 0:
                block = EncoderBlock(6, filters * 2, apply_batchnorm=False)
            else:
                block = EncoderBlock(filters, filters * 2, apply_batchnorm=True)
            self.encoder.append(block)
            filters = filters * 2

        self.conv1 = nn.Conv2d(256, 512,  kernel_size=4, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, kernel_size=4, stride=1)

    def forward(self, fake:torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fake, x], dim=1)

        for block in self.encoder:
            x = block(x)

        x = nn.ZeroPad2d(1)(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = nn.ZeroPad2d(1)(x)
        x = self.conv2(x)

        return x



