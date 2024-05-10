from __future__ import annotations
from typing import Union, Callable, Dict
from torch import nn
from torch.optim import Adam
import torch
from modules import (
    GeneratorLoss,
    DiscriminatorLoss,
    UNet,
    AttentionUNet,
    ConditionalDiscriminator,
)
from models.segmenter import SemanticSegmenter
from utils import GenerationMetric
from model import FacadesModel
import segmentation_models_pytorch as smp


class AdversarialTranslator(FacadesModel):
    METRIC = GenerationMetric
    NAME = "adversarial"

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        segmenter: SemanticSegmenter,
        device: str,
        weights: bool = False,
    ):
        super().__init__(device)
        self.generator = generator
        self.discriminator = discriminator
        self.segmenter = segmenter
        self.gen_loss = GeneratorLoss(weights=weights)
        self.dis_loss = DiscriminatorLoss()
        self.seg_loss = nn.BCEWithLogitsLoss()

    def train(self, *args, opt: Callable = Adam, lr: float = 2e-4, **kwargs):
        """Override the general train function to specify the generator and discriminator optimizers.

        Args:
            opt (Callable, optional): General optimizer. Defaults to Adam.
            lr (float, optional): Learning rate. Defaults to 2e-4.
        """
        self.gen_optimizer = opt(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.dis_optimizer = opt(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        super().train(*args, **kwargs)

    def forward(
        self, reals: torch.Tensor, masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # generator loss
        fakes = self.generator(masks)
        fake_preds = self.discriminator(masks, fakes)
        gen_loss = self.gen_loss(fakes, reals, fake_preds)

        # discriminator loss
        fake_dis = self.discriminator(masks, fakes.detach())
        real_dis = self.discriminator(masks, reals)
        dis_loss = self.dis_loss(fake_dis, real_dis)

        # segmenter loss
        fake_segmented = self.segmenter.model(fakes)
        real_segmented = self.segmenter.segment(masks)
        seg_loss = self.seg_loss(fake_segmented, real_segmented)

        return {"gen_loss": gen_loss, "dis_loss": dis_loss, "seg_loss": seg_loss}

    def backward(
        self, gen_loss: torch.Tensor, dis_loss: torch.Tensor, seg_loss: torch.Tensor
    ):
        # generator params update
        # gen_loss += seg_loss/4
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()
        self.optimize = 1
        # discriminator params update
        self.dis_optimizer.zero_grad()
        dis_loss.backward()
        self.dis_optimizer.step()

    @torch.no_grad()
    def eval_step(self, reals: torch.Tensor, masks: torch.Tensor) -> GenerationMetric:
        labels = self.segmenter.label(masks)
        fakes = self.generator(masks)
        reals_segmented = self.segmenter.label(self.segmenter.model(reals))
        fakes_segmented = self.segmenter.label(self.segmenter.model(fakes))
        return GenerationMetric(
            fakes.flatten(),
            fakes_segmented.flatten(),
            reals.flatten(),
            reals_segmented.flatten(),
            labels.flatten(),
        )

    @torch.no_grad()
    def pred_step(self, reals: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        fakes = self.generator(masks)
        fakes_segmented = self.segmenter.segment(self.segmenter.model(fakes))
        return (
            torch.cat(
                [reals, self.segmenter.segment(masks), fakes, fakes_segmented], -1
            )
            * 0.5
            + 0.5
        )

    def save(self, path: str):
        torch.save(self.generator, f"{path}/generator.pt")
        torch.save(self.discriminator, f"{path}/discriminator.pt")
        self.segmenter.save(path)

    @classmethod
    def load(cls, path: str, device: str) -> AdversarialTranslator:
        generator = torch.load(f"{path}/generator.pt").to(device)
        discriminator = torch.load(f"{path}/discriminator.pt").to(device)
        segmenter = SemanticSegmenter.load(path, device)
        return cls(generator, discriminator, segmenter, device)

    @classmethod
    def build(
        cls,
        segmenter: Union[str, SemanticSegmenter],
        gen_type: str = "base",
        pretrained: str = "resnext50_32x4d",
        device: str = "cuda:0",
        weights: bool = False,
    ):
        """Build an adversarial image-to-image translator.

        Args:
            segmenter (Union[str, SemanticSegmenter]): Semantic segmentation model for evaluation.
            num_blocks (int, optional): Number of blocks of the PathGAN discriminator. Defaults to 4.
            generator (str): Type of UNet generator (base, deformable or attention-based).
            device (_type_, optional): _description_. Defaults to 'cuda:0'.

        Returns:
            _type_: _description_
        """
        if isinstance(segmenter, str):
            segmenter = SemanticSegmenter.load(segmenter, device)
        if gen_type == "base":
            generator = UNet(3, 3).to(device)
        elif gen_type == "deform":
            generator = UNet(3, 3, conv="deform").to(device)
        elif gen_type == "attn":
            generator = AttentionUNet(3, 3)
        elif gen_type == "link":
            generator = smp.Linknet(pretrained, in_channels=3, classes=3)
        elif gen_type == "fpn":
            generator = smp.FPN(
                pretrained,
                in_channels=3,
                classes=3,
                decoder_pyramid_channels=512,
                decoder_segmentation_channels=512,
                activation=nn.Tanh,
                decoder_dropout=0.5,
            )
        elif gen_type == "psp":
            generator = smp.PSPNet(
                pretrained,
                in_channels=3,
                classes=3,
                encoder_depth=5,
                activation=nn.Tanh,
                upsampling=32,
                psp_dropout=0,
            )
        else:
            raise NotImplementedError(f"The generator type {gen_type} is not avaiable")
        discriminator = ConditionalDiscriminator(3, conv="base")
        print(generator)
        return AdversarialTranslator(
            generator.to(device),
            discriminator.to(device),
            segmenter,
            device,
            weights=weights,
        )
