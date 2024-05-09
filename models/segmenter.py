from __future__ import annotations
from model import FacadesModel
from typing import Tuple, Callable, Dict, Union, Optional , List 
from torch.optim import Adam, RMSprop
from torch import nn 
import torch, contextlib
import numpy as np 
from utils import FacadesDataset, SegmentationMetric
from modules import VGGNet, FCN8s
from kmeans_pytorch import kmeans, kmeans_predict
from torch.nn.functional import sigmoid

class SemanticSegmenter(FacadesModel):
    METRIC = SegmentationMetric
    N_KMEANS = 100
    
    def __init__(self, model: nn.Module, centers: torch.Tensor, device: str):
        super().__init__(device)
        self.centers = centers
        self.model = model 
        self.loss = nn.BCEWithLogitsLoss()
        
    def train(self, train: FacadesDataset, *args, opt: Callable = Adam, lr: float = 1e-3, **kwargs):
        # train kmeans 
        train.transform = self.TRANSFORM
        points = torch.cat([train[i][1].flatten(1,2) for i in range(self.N_KMEANS)], 1).permute(1,0)
        _, self.centers = kmeans(X=points, num_clusters=self.centers.shape[0], device=self.device, distance='euclidean')
        self.optimizer = opt(self.model.parameters(), lr=lr)
        super().train(train, *args, **kwargs)
        
    def forward(self, reals: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        segmented = self.segment(masks)
        outputs = self.model(reals)
        loss = self.loss(outputs.flatten(), sigmoid(segmented.flatten()))
        return {'loss': loss}
    
    def backward(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def eval_step(self, reals: torch.Tensor, masks: torch.Tensor)  -> SegmentationMetric:
        labeled = self.label(masks)
        outputs = self.label(self.model(reals))
        return SegmentationMetric(outputs.flatten(), labeled.flatten())
    
    @torch.no_grad()
    def pred_step(self, reals: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        segmented = self.segment(masks)
        preds = self.segment(self.model(reals))
        return torch.cat([reals, masks, segmented, preds], -1)*0.5+0.5

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        with contextlib.redirect_stdout(None):
            segments = []
            for img in x.unbind(0):
                labs = kmeans_predict(img.flatten(1,2).T, self.centers, 'euclidean', device=self.device)
                segment = self.centers[labs].reshape(*img.shape[-2:], -1).permute(2, 0, 1)
                segments.append(segment)
        return torch.stack(segments, 0).to(self.device)
    
    def label(self, x: torch.Tensor) -> torch.Tensor:
        with contextlib.redirect_stdout(None):
            labels = []
            for img in x.unbind(0):
                labs = kmeans_predict(img.flatten(1,2).T, self.centers, 'euclidean', device=self.device)
                labeled = labs.reshape(*img.shape[-2:])
                labels.append(labeled)
        return torch.stack(labels, 0).to(self.device)
    
    def save(self, path: str):
        torch.save(self.centers, f'{path}/centers.pt')
        torch.save(self.model, f'{path}/model.pt')
    
    @classmethod
    def load(cls, path: str, device: str) -> SemanticSegmenter:
        model = torch.load(f'{path}/model.pt', map_location=device).to(device)
        centers = torch.load(f'{path}/centers.pt', map_location=device).to(device)
        return SemanticSegmenter(model, centers, device)
    
    @classmethod
    def build(cls, n_labels: int = 5, device: str = 'cuda:0'):
        # random centers 
        centers = torch.zeros(n_labels, 3)
        vgg_model = VGGNet(requires_grad=True).to(device)
        model = FCN8s(3, vgg_model).to(device)
        return cls(model, centers, device)
    