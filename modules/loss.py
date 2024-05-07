import torch
import torch.nn as nn
from utils.fn import gaussian2d
from torchvision.transforms.functional import rgb_to_grayscale
from torch.nn.functional import max_pool2d
from typing import Optional


class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100, weights: bool = True, k1: int = 17, k2: int = 11):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.loss = nn.BCEWithLogitsLoss()
        self.weights = weights
        if weights:
            self.mask = gaussian2d(256, 256).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        self.k1, self.k2 = k1, k2
        

    def forward(
            self, 
            fakes: torch.Tensor, 
            reals: torch.Tensor, 
            fake_preds: torch.Tensor
        ):
        dis_loss = self.loss(fake_preds, torch.ones_like(fake_preds))
        l1_loss = self.l1(fakes, reals)
        if self.weights:
            gray_reals = rgb_to_grayscale(reals*0.5+0.5)
            W = max_pool2d(gray_reals, kernel_size=self.k1, stride=1, padding=self.k1//2) \
                + max_pool2d(-gray_reals, kernel_size=self.k1, stride=1, padding=self.k1//2)
            W += self.mask.to(fakes.device)
            W = max_pool2d(W, kernel_size=self.k2, stride=1, padding=self.k2//2)
            W = (W-W.min())/(W.max()-W.min())
            mae = W*((fakes-reals).abs())
            l1_loss = mae.mean()
        return dis_loss + (self.alpha*l1_loss)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_target = self.loss(fake_pred, torch.zeros_like(fake_pred))
        real_target = self.loss(real_pred, torch.ones_like(real_pred))
        loss = fake_target + real_target
        return loss / 2


        
    