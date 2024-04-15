import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha=alpha
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()
        
    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        gan_loss = self.bce(fake_pred, fake_target)
        l1_loss =  self.l1(fake, real)
        
        loss = gan_loss + (self.alpha * l1_loss)

        return loss
    
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn=nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_target = self.loss_fn(fake_pred, torch.zeros_like(fake_pred))
        real_target = self.loss_fn(real_pred, torch.ones_like(real_pred))

        loss = fake_target + real_target

        return loss