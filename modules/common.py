from torch import nn 
from typing import Optional
import torch 
from torchvision.ops import deform_conv2d

class ConvBlock(nn.Module):
    """Implementation of a convolutional block with batch normalization, dropout and activation."""
    
    def __init__(
            self, 
            n_in: int, 
            n_out: Optional[int] = None, 
            k: int = 3,
            batch_norm: bool = True,
            act: nn.Module = nn.LeakyReLU(0.1),
            dropout: float = 0.0,
            n_layers: int = 1,
            **kwargs
        ):
        super().__init__()
        
        if n_layers == 1:
            self.conv = nn.Conv2d(in_channels=n_in, out_channels=n_out or n_in, kernel_size=k, **kwargs)
        else:
            self.conv = nn.ModuleList()
            last = n_in
            for _ in range(n_layers):
                self.conv.append(nn.Conv2d(in_channels=last, out_channels=n_out or last, kernel_size=k, **kwargs))
                last = n_out 
        self.bn = nn.BatchNorm2d(n_out or n_in)
        self.act = act 
        self.drop = nn.Dropout2d(dropout, inplace=True)
        self.batch_norm = batch_norm
        self.n_layers = n_layers
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.drop(x)
        return self.act(x)
    
    


        
class TransposedConvBlock(nn.Module):
    """Implementation of a tranposed convolutional block with batch normalization, dropout and activation."""
    
    def __init__(
            self, 
            n_in: int, 
            n_out: Optional[int] = None, 
            k: int = 3,
            batch_norm: bool = True,
            act: nn.Module = nn.LeakyReLU(0.1),
            dropout: float = 0.0,
            **kwargs
        ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=n_in, out_channels=n_out or n_in, kernel_size=k, **kwargs)
        self.bn = nn.BatchNorm2d(n_out or n_in)
        self.act = act
        self.drop = nn.Dropout2d(dropout, inplace=True)
        self.batch_norm = batch_norm
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.drop(x)
        return self.act(x)
    



class DeformableConvBlock(nn.Module):
    """Implementation of the deformable convolution with regularization at https://arxiv.org/abs/1811.11168 
    adapted from https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2."""
    
    def __init__(
            self, 
            n_in: int, 
            n_out: Optional[int] = None, 
            k: int = 3,
            batch_norm: bool = True,
            act: nn.Module = nn.LeakyReLU(0.1),
            dropout: float = 0.0,
            padding: int = 0,
            stride: int = 1,
            **kwargs
        ):
        super().__init__()
        self.offset_conv = nn.Conv2d(n_in, 2 * k * k, kernel_size=k, padding=padding, stride=stride, **kwargs)
        self.modulator_conv = nn.Conv2d(n_in, 1 * k * k, kernel_size=k, padding=padding, stride=stride, **kwargs)
        self.regular_conv = nn.Conv2d(n_in, n_out or n_in, k, padding=padding, stride=stride, **kwargs)
        self.bn = nn.BatchNorm2d(n_out or n_in)
        self.act = act
        self.drop = nn.Dropout2d(dropout, inplace=True)
        self.batch_norm = batch_norm
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.
        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = deform_conv2d(
            input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, 
            padding=self.padding, mask=modulator, stride=self.stride
        )
        
        if self.batch_norm:
            x = self.bn(x)
        x = self.drop(x)
        return self.act(x)
    
    

    
class AttentionBlock(nn.Module):
    def __init__(
            self, 
            n_in1: int, 
            n_in2: int, 
            n_out: int, 
            batch_norm: bool = True, 
            dropout: float = 0.0, 
            act: nn.Module = nn.LeakyReLU()
        ):
        super().__init__()
        
        self.Q = nn.Conv2d(n_in1, n_in1, 1)
        self.K = nn.Conv2d(n_in2, n_in2, 1)
        self.S = nn.Conv2d(n_in1+n_in2, n_out, 1)
        self.V = nn.Conv2d(n_in2, n_out, 1)
        self.bn = nn.BatchNorm2d(n_in2)
        self.batch_norm = batch_norm
        self.act = act 
        self.drop = nn.Dropout2d(dropout, inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """2-dimensional attention

        Args:
            x1 (torch.Tensor): ``[batch_size, n_in1, height, width]``
            x2 (torch.Tensor): ``[batch_size, n_in2, heigh, width]``.

        Returns:
            torch.Tensor: Attention representaiton.
        """
        Q, K, V = self.Q(x1), self.K(x2), self.V(x2)
        S = self.S(torch.cat([Q, K], 1))
        out = V*S 
        if self.batch_norm:
            out = self.bn(out)
        out = self.drop(out)
        out = self.act(out)
        return out 
    
    