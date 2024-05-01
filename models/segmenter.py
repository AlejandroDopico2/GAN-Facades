from __future__ import annotations
from model import VisionModel
from typing import Tuple, Callable, Dict, Union, Optional , List 
from torch.optim import Adam, RMSprop
from torch import nn 
import torch, contextlib
import numpy as np 
from utils import FacadesDataset, SegmentationMetric
from modules import VGGNet, FCN8s
from kmeans_pytorch import kmeans, kmeans_predict
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda 
from PIL import Image

class SemanticSegmenter(VisionModel):
    METRIC = SegmentationMetric
    N_KMEANS = 100
    TRANSFORM = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    def __init__(self, model: nn.Module, centers: torch.Tensor, device: str):
        super().__init__(device)
        self.centers = centers
        self.model = model 
        self.loss = nn.CrossEntropyLoss()
        
        
    def train(self, train: FacadesDataset, *args, opt: Callable = RMSprop, lr: float = 1e-4, **kwargs):
        # train kmeans 
        train.transform = self.TRANSFORM 
        points = torch.stack([train[i][1] for i in range(self.N_KMEANS)], 0).permute(0, 2, 3, 1).flatten(0, -2)
        _, centers = kmeans(X=points, num_clusters=self.centers.shape[0], device=self.device, distance='cosine')
        self.centers = centers 
        self.optimizer = opt(self.model.parameters(), lr=lr)
        super().train(train, *args, **kwargs)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        labeled = self.get_labels(targets)
        outputs = self.model(inputs)
        loss = self.loss(outputs.permute(0, 2, 3, 1).reshape(-1, self.centers.shape[0]), labeled.flatten())
        return {'loss': loss}
    
    def backward(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0, norm_type=2)
        self.optimizer.step()

    @torch.no_grad()
    def eval_step(self, inputs: torch.Tensor, targets: torch.Tensor)  -> SegmentationMetric:
        labeled = self.get_labels(targets)
        outputs = self.pred_step(inputs, targets)
        return SegmentationMetric(outputs, labeled)
    
    @torch.no_grad()
    def pred_step(self, inputs: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).argmax(1)

    def get_labels(self, targets: torch.Tensor) -> torch.Tensor:
        with contextlib.redirect_stdout(None):
            labeled = kmeans_predict(targets.permute(0, 2, 3, 1).flatten(0, -2), self.centers, 'cosine', device=self.device)
        labeled = labeled.reshape(targets.shape[0], *targets.shape[-2:]).to(self.device)
        return labeled
    
    def save(self, path: str):
        torch.save(self.centers, f'{path}/centers.pt')
        torch.save(self.model, f'{path}/model.pt')
    
    @classmethod
    def load(cls, path: str, device: str) -> SemanticSegmenter:
        model = torch.load(f'{path}/model.pt').to(device)
        centers = torch.load(f'{path}/centers.pt').to(device)
        return SemanticSegmenter(model, centers, device)
    
    @classmethod
    def build(
            cls, 
            n_labels: int = 4,
            device: str = 'cuda:0'
        ):
        # random centers 
        centers = torch.empty(n_labels, 3)
        vgg_model = VGGNet(requires_grad=True).to(device)
        model = FCN8s(n_labels, vgg_model).to(device)
        return cls(model, centers, device)
    