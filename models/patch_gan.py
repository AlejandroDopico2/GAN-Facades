import torch
import torch.nn as nn

from models.common import EncoderBlock


class Discriminator(nn.Module):
    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()

        filters = 32
        max_filters = 512

        self.encoder = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                block = EncoderBlock(
                    6, filters * 2, apply_batchnorm=False, negative_slope=0.2
                )
            else:
                block = EncoderBlock(
                    filters, filters * 2, apply_batchnorm=True, negative_slope=0.2
                )
            self.encoder.append(block)
            filters = filters * 2

        self.conv2 = nn.Conv2d(filters, 1, kernel_size=4, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, fake], dim=1)

        for block in self.encoder:
            x = block(x)

        # x = nn.ZeroPad2d(1)(x)
        # x = self.conv1(x)
        # x = self.bn(x)
        # x = self.relu(x)
        # x = nn.ZeroPad2d(1)(x)
        x = self.conv2(x)

        return self.sigmoid(x)


if __name__ == "__main__":
    net = Discriminator(4)

    x = torch.rand(1, 3, 256, 256)
    fake = torch.rand(1, 3, 256, 256)

    x = net(fake, x)
    print(net)
    print(f"x {x.shape}, fake {fake.shape}")
