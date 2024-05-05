from __future__ import annotations
from typing import Tuple, List, Dict, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch, os
from utils import FacadesDataset, Metric
import numpy as np 
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomAffine, Normalize, Lambda

class FacadesModel:
    """Abstract representation of a Facades Model that implements the general train, predict and 
    evaluation methods."""
    
    # specify evaluation metric and data transformation
    METRIC = None
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = device 
        self.TRANSFORM = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        # self.TRANSFORM = Compose([ToTensor(), Lambda(lambda x: (x-x.min()/(x.max()-x.min())))])
        
    def train(
            self,
            train: FacadesDataset,
            dev: FacadesDataset,
            path: str, 
            epochs: int = 100,
            batch_size: int = 20, 
            train_patience: int = 20,
            dev_patience: int = 10,
            aug: bool = False
        ) -> Metric:
        """Model training with:
        - Early stopping over the train and dev set.
        - Prediction of the validation set.

        Args:
            train (FacadesDataset): Train set.
            dev (FacadesDataset): Validation set.
            path (str): Folder to store PyTorch modules.
            epochs (int, optional): Training epochs. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 20.
            train_patience (int, optional): Number of allowed epochs with no train improvement. Defaults to 20.
            dev_patience (int, optional): Number of allowed epochs with no validation improvement. Defaults to 10.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(f'{path}/preds/'):
            os.makedirs(f'{path}/preds/')
        if aug:
            train.transform = Compose(self.TRANSFORM.transforms + [
                RandomHorizontalFlip(0.5)
            ])
        else:
            train.transform = self.TRANSFORM
        train_dl = DataLoader(train, batch_size, shuffle=True)
        
        best_metric, train_loss, train_improv, dev_improv = self.METRIC(), np.Inf, train_patience, dev_patience
        best_metric._loss = np.Inf
        for epoch in range(1, epochs+1):
            with tqdm(desc='train', total=len(train)) as bar:
                for reals, masks in train_dl:
                    losses = self.forward(reals.to(self.device), masks.to(self.device))
                    self.backward(**losses)
                    bar.update(reals.shape[0])
                    bar.set_postfix({name: round(float(value.detach()), 2) for name, value in losses.items()})
                    
            dev_metric = self.evaluate(dev, batch_size)
            self.predict(dev, f'{path}/preds', batch_size)

            if dev_metric.improves(best_metric):
                print(f'Epoch {epoch}: (improved)')
                dev_improv = dev_patience
                best_metric = dev_metric
                self.save(path)
            else:
                print(f'Epoch {epoch}:')
                dev_improv -= 1
            print(f'[dev]: {repr(dev_metric)}')
            if sum(losses.values()) < train_loss:
                train_improv = train_patience
                train_loss = sum(losses.values())
            else:
                train_improv -= 1
                
            if train_improv == 0:
                print('No improvement in the train set')
                break 
            if dev_improv == 0:
                print('No improvement in the dev set')
                break 
                
        # save dev inputs 
        self = self.__class__.load(path, self.device)
        best_metric.save(f'{path}/result.pickle')
        return best_metric

    
    @torch.no_grad()
    def evaluate(self, data: FacadesDataset, batch_size: int) -> Metric:
        """Evaluates the input dataset.

        Args:
            data (FacadesDataset): Input dataset.
            batch_size (int): Batch size for model inference.

        Returns:
            Metric: Evaluation metric.
        """
        data.transform = self.TRANSFORM
        data_dl = DataLoader(data, batch_size, shuffle=False)
        metric = self.METRIC()
        for inputs, targets in tqdm(data_dl, total=len(data_dl), desc='eval'):
            metric += self.eval_step(inputs.to(self.device), targets.to(self.device))
        return metric 
    
    @torch.no_grad()
    def predict(self, data: FacadesDataset, path: str, batch_size: int):
        """Stores the output images of the model.
        
        Args:
            data (FacadesDataset): Input dataset.
            path (str): Folder to store output images.
            batch_size (int): Batch size for model inference.
        """
        data.transform = self.TRANSFORM
        data_dl = DataLoader(data, batch_size, shuffle=False)
        preds = []
        for reals, masks in tqdm(data_dl, total=len(data_dl), desc='predict'):
            reals, masks = reals.to(self.device), masks.to(self.device)
            preds += self.pred_step(reals, masks).unbind(0)
        for img_path, pred in zip(data.image_paths, preds):
            filename = img_path.split('/')[-1]
            save_image(pred, f'{path}/{filename}')
            
    """Specific methods that must be implemented in inherited models."""
            
    def forward(self, imgs: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def backward(self, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def eval_step(self, reals: torch.Tensor, masks: torch.Tensor) -> Metric:
        raise NotImplementedError
    
    @torch.no_grad()
    def pred_step(self, reals: torch.Tensor, masks: torch.Tensor): 
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: str, device: str) -> FacadesModel:
        raise NotImplementedError
