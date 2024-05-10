from __future__ import annotations
from model import FacadesModel
from typing import Callable, Dict
from torch.optim import Adam
from torch import nn 
import torch, contextlib
import numpy as np 
from utils import FacadesDataset, SegmentationMetric
from modules import VGGNet, FCN8s
from kmeans_pytorch import kmeans, kmeans_predict
from torch.nn.functional import sigmoid

class SemanticSegmenter(FacadesModel):
    METRIC = SegmentationMetric
    NAME = 'segmenter'
    N_KMEANS = 100 # number of samples to optimize kmeans
    
    def __init__(self, model: nn.Module, centers: torch.Tensor, device: str):
        """Initialize the semantic segmentation network.

        Args:
            model (nn.Module): Neural model.
            centers (torch.Tensor): k-Means centers.
            device (str): CUDA device.
        """
        super().__init__(device)
        self.centers = centers
        self.model = model 
        self.loss = nn.BCEWithLogitsLoss()
        
    def train(self, train: FacadesDataset, *args, opt: Callable = Adam, lr: float = 1e-3, **kwargs):
        """Override the general train function to train the k-Means algorithm.

        Args:
            opt (Callable, optional): General optimizer. Defaults to Adam.
            lr (float, optional): Learning rate. Defaults to 2e-4.
        """
        # train kmeans 
        train.transform = self.TRANSFORM
        points = torch.cat([train[i][1].flatten(1,2) for i in range(self.N_KMEANS)], 1).permute(1,0)
        _, self.centers = kmeans(X=points, num_clusters=self.centers.shape[0], device=self.device, distance='euclidean')
        self.optimizer = opt(self.model.parameters(), lr=lr)
        super().train(train, *args, **kwargs)
        
    def forward(self, reals: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            reals (torch.Tensor): Real images.
            masks (torch.Tensor): Map images.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of model losses.
        """
        segmented = self.segment(masks)
        outputs = self.model(reals)
        loss = self.loss(outputs.flatten(), sigmoid(segmented.flatten()))
        return {'loss': loss}
    
    def backward(self, loss: torch.Tensor):
        """Backward pass.

        Args:
            loss (torch.Tensor): Model loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def eval_step(self, reals: torch.Tensor, masks: torch.Tensor)  -> SegmentationMetric:
        """Evaluation step.

        Args:
            reals (torch.Tensor): Real images.
            masks (torch.Tensor): Map images.
            
        Returns:
            SegmentationMetric: Segmentation metrics.
        """
        labeled = self.label(masks)
        outputs = self.label(self.model(reals))
        return SegmentationMetric(outputs.flatten(), labeled.flatten())
    
    @torch.no_grad()
    def pred_step(self, reals: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Prediction step. 

        Args:
            reals (torch.Tensor): Real images.
            masks (torch.Tensor): Map images.

        Returns:
            torch.Tensor: Concatenation of real, map, segmented and and segmented prediction images.
        """
        segmented = self.segment(masks)
        preds = self.segment(self.model(reals))
        return torch.cat([reals, masks, segmented, preds], -1)*0.5+0.5

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the segmentation of a continuous map by approximating each pixel to its nearest 
        center.

        Args:
            x (torch.Tensor): ``[batch_size, 3, height, width]``.

        Returns:
            torch.Tensor: ``[batch_size, 3, height, width]``. Segmented images.
        """
        with contextlib.redirect_stdout(None):
            segments = []
            for img in x.unbind(0):
                labs = kmeans_predict(img.flatten(1,2).T, self.centers, 'euclidean', device=self.device)
                segment = self.centers[labs].reshape(*img.shape[-2:], -1).permute(2, 0, 1)
                segments.append(segment)
        return torch.stack(segments, 0).to(self.device)
    
    def label(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the labels of a continuous map by approximating each pixel to its nearest center 
        and labeling it to an integer.

        Args:
            x (torch.Tensor): ``[batch_size, 3, height, width]``.

        Returns:
            torch.Tensor: ``[batch_size, height, width]``. Labeled images.
        """
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
        """Builds a segmentation model.

        Args:
            n_labels (int, optional): Number of labels. Defaults to 5.
            device (_type_, optional): CUDA device. Defaults to 'cuda:0'.
        """
        centers = torch.zeros(n_labels, 3)
        vgg_model = VGGNet(requires_grad=True).to(device)
        model = FCN8s(3, vgg_model).to(device)
        return cls(model, centers, device)
    