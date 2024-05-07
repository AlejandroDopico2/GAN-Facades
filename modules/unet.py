import torch
import torch.nn as nn
from modules.common import ConvBlock, TransposedConvBlock, DeformableConvBlock, AttentionBlock


class UNet(nn.Module):
    """PyTorch implementation of the U-Net proposed in the Pix2Pix model at https://arxiv.org/abs/1611.07004."""
    
    KERNEL_SIZE = 4
    STRIDE = 2
    ENCODER = [64, 128, 256, 512, 512, 512, 512]
    DECODER = [512, 512, 512, 256, 128, 64]

    def __init__(self, n_in: int, n_out: int, conv: str = 'base'):
        super().__init__()

        self.encoder = nn.ModuleList()
        last = n_in
        for i, n in enumerate(self.ENCODER):
            params = dict(k=self.KERNEL_SIZE, batch_norm=(i != 0), stride=self.STRIDE, act=nn.LeakyReLU(0.2), padding=1, bias=False)
            if conv == 'base':
                block = ConvBlock(last, n, **params)
            elif conv == 'deform':
                block = DeformableConvBlock(last, n, **params)
            else:
                raise NotImplementedError
            self.encoder.append(block)
            last = n 

        params = dict(k=self.KERNEL_SIZE, stride=self.STRIDE, batch_norm=True, act=nn.LeakyReLU(0.2), dropout=0.5, padding=1, bias=False)
        self.decoder = nn.ModuleList(
            [TransposedConvBlock(last, self.DECODER[0], **params)]
        )
        last = self.DECODER[0]
        for j, n in zip(range(2, len(self.DECODER)+1), self.DECODER[1:]):
            block = TransposedConvBlock(self.ENCODER[-j]+last, n, **params)
            self.decoder.append(block)
            last = n 
        self.out = TransposedConvBlock(last+self.ENCODER[0], n_out, self.KERNEL_SIZE, stride=self.STRIDE, padding=1, bias=False)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x)
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.out(x))
        return x 
    

    
class AttentionUNet(nn.Module):
    
    KERNEL_SIZE = 4
    STRIDE = 2
    ENCODER = [64, 128, 256, 512, 512, 512, 512]
    DECODER = [512, 512, 512, 256, 128, 64]

    def __init__(self, n_in: int, n_out: int, conv: str = 'base'):
        super().__init__()

        self.encoder = nn.ModuleList()
        last = n_in
        for i, n in enumerate(self.ENCODER):
            params = dict(k=self.KERNEL_SIZE, batch_norm=(i != 0), stride=self.STRIDE, act=nn.LeakyReLU(0.2), padding=1)
            if conv == 'base':
                block = ConvBlock(last, n, **params)
            elif conv == 'deform':
                block = DeformableConvBlock(last, n, **params)
            else:
                raise NotImplementedError
            self.encoder.append(block)
            last = n 
            
        self.decoder = nn.ModuleList(
            [TransposedConvBlock(last, self.DECODER[0], self.KERNEL_SIZE, stride=self.STRIDE, batch_norm=True, act=nn.ReLU(), dropout=0.5, padding=1)]
        )
        last = self.DECODER[0]
        self.attns = nn.ModuleList([AttentionBlock(last, self.ENCODER[-1], last)])
        for j, n in zip(range(2, len(self.DECODER)+1), self.DECODER[1:]):
            block = TransposedConvBlock(last, n, self.KERNEL_SIZE, batch_norm=True, stride=self.STRIDE, act=nn.ReLU(), dropout=0.5, padding=1)
            self.decoder.append(block)
            self.attns.append(AttentionBlock(n, self.ENCODER[-j-1], n))
            last = n 
        self.out = TransposedConvBlock(last, n_out, k=self.KERNEL_SIZE, stride=self.STRIDE, padding=1)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        for block, skip, attn in zip(self.decoder, reversed(skips[:-1]), self.attns):
            x = block(x)
            x = attn(x, skip)
        x = self.act(self.out(x))
        return x 
    