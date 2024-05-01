from __future__ import annotations
from typing import Union, Callable, Dict
from torch import nn 
from torch.optim import Adam
import torch 
from models.loss import GeneratorLoss, DiscriminatorLoss
from models.segmenter import SemanticSegmenter
from utils import GenerationMetric, FacadesDataset
from model import VisionModel
from modules import UNet, Discriminator
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm 

class AdversarialTranslator(VisionModel):
    METRIC = GenerationMetric
    NAME = 'adversarial'
    TRANSFORM = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            segmenter: SemanticSegmenter,
            device: str
        ):
        super().__init__(device)
        self.generator = generator 
        self.discriminator = discriminator 
        self.segmenter = segmenter
        self.gen_loss = GeneratorLoss(alpha=100)
        self.dis_loss = DiscriminatorLoss()
        
        
    def train(self, *args, opt: Callable = Adam, lr: float = 2e-4, **kwargs):
        """Override the general train function to specify the generator and discriminator optimizers.

        Args:
            opt (Callable, optional): General optimizer. Defaults to Adam.
            lr (float, optional): Learning rate. Defaults to 2e-4.
        """
        self.gen_optimizer = opt(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.dis_optimizer = opt(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        super().train(*args, **kwargs)
                
    def forward(self, real: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # generator loss 
        fake = self.generator(mask)
        fake_pred = self.discriminator(fake, mask)
        gen_loss = self.gen_loss(fake, real, fake_pred)
        
        # discriminator loss 
        fake = self.generator(mask).detach()
        fake_pred = self.discriminator(fake, mask)
        real_pred = self.discriminator(real, mask)
        dis_loss = self.dis_loss(fake_pred, real_pred)
        
        return {'gen_loss': gen_loss, 'dis_loss': dis_loss}
    
    def backward(self, gen_loss: torch.Tensor, dis_loss: torch.Tensor):
        # generator params update
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()
        
        # discrimnator params update 
        self.dis_optimizer.zero_grad()
        dis_loss.backward()
        self.dis_optimizer.step()
        
    @torch.no_grad()
    def eval_step(self, real: torch.Tensor, mask: torch.Tensor) -> GenerationMetric:
        labels = self.segmenter.get_labels(mask)
        fakes = self.generator(mask)
        real_segmented = self.segmenter.pred_step(real, None)
        fake_segmented = self.segmenter.pred_step(fakes, None)
        return GenerationMetric(real_segmented, fake_segmented, labels)
    
    @torch.no_grad()
    def pred_step(self, _: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fakes = self.generator(mask)
        return fakes

    def save(self, path: str):
        torch.save(self.generator, f'{path}/generator.pt')
        torch.save(self.discriminator, f'{path}/discriminator.pt')
        torch.save(self.segmenter, f'{path}/segmenter.pt')

    @classmethod
    def load(cls, path: str, device: str) -> AdversarialTranslator:
        generator = torch.load(f'{path}/generator.pt')
        discriminator = torch.load(f'{path}/discriminator.pt')
        segmenter = torch.load(f'{path}/segmenter.pt')
        return cls(generator, discriminator, segmenter, device)
    
    @classmethod
    def build(cls, segmenter: Union[str, SemanticSegmenter], device: str):
        if isinstance(segmenter, str):
            segmenter = SemanticSegmenter.load(segmenter, device)
        generator = UNet(num_blocks=8, filter_size=4).to(device)
        discriminator = Discriminator().to(device)
        return AdversarialTranslator(generator, discriminator, segmenter, device)
        
                
        