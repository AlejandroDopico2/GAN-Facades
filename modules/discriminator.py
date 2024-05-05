import torch
import torch.nn as nn
from modules.common import ConvBlock, DeformableConvBlock


class ConditionalDiscriminator(nn.Module):
    N_FILTERS = [32, 64]
    KERNEL_SIZE = 4
    STRIDE = 2
    
    def __init__(self, n_in: int = 3, conv: str = 'base') -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        last = n_in*2
        for i, n in enumerate(self.N_FILTERS):
            params = dict(k=self.KERNEL_SIZE, stride=self.STRIDE, batch_norm=(i!=0), act=nn.LeakyReLU(0.2), padding=1)
            if conv == 'base':
                block = ConvBlock(last, n, **params)
            elif conv == 'deform':
                block = DeformableConvBlock(last, n, **params)
            else:
                raise NotImplementedError
            self.encoder.append(block)
            last = n
        self.conv = nn.Conv2d(last, 1, kernel_size=4, stride=self.STRIDE, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, masks: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masks, imgs], dim=1)
        for block in self.encoder:
            x = block(x)
        x = self.conv(x)
        return self.sigmoid(x)

