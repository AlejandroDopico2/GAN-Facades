import torch
import torch.nn as nn

from modules.common import DecoderBlock, EncoderBlock


class UNet(nn.Module):
    def __init__(
            self, 
            num_blocks: int, 
            filter_size: int, 
            n_classes: int = 3,
            in_channels: int = 64,
            out_channels: int = 128,
            activation: nn.Module = nn.Tanh()
        ) -> None:
        super().__init__()

        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Conv2d(3, 64, kernel_size=filter_size, stride=2, padding=1, bias=False)
        )
        for _ in range(num_blocks - 2):
            self.encoder.append(EncoderBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)

        self.encoder.append(
            EncoderBlock(in_channels, out_channels, apply_batchnorm=False)
        )

        self.decoder = nn.ModuleList()
        self.decoder.append(DecoderBlock(in_channels, out_channels, apply_dropout=True))
        for i in range(num_blocks - 2):
            self.decoder.append(
                DecoderBlock(
                    2 * in_channels,
                    out_channels,
                    apply_dropout=True if i < 2 else False,
                )
            )

            if i > 1:
                in_channels = out_channels
                out_channels = in_channels // 2

        # Final convolutional layer in the decoder
        self.decoder.append(
            nn.ConvTranspose2d(
                2 * in_channels,
                n_classes,
                kernel_size=filter_size,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for block, skip in zip(self.decoder[:-1], skips):
            x = block(x)
            x = torch.cat([x, skip], dim=1)
            # print(f"x {x.shape}, skip {skip.shape}")
        x = self.decoder[-1](x)
        return self.activation(x)


if __name__ == "__main__":
    net = UNet(8, 4)

    x = torch.rand(1, 3, 256, 256)

    x = net(x)
