from __future__ import annotations
from typing import Optional
from torch import nn 
import torch 


class Metric:
    METRICS = None 
    ATTRIBUTES = None 
    KEY = None
    SCALE = None
    MODE = None

    def __init__(self, *args, **kwargs):
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, 0.0)
        self.n = 0
        
        if len(args) > 0 or len(kwargs) > 0:
            self(*args, **kwargs)
    
    def __call__(self, pred: torch.Tensor, gold: torch.Tensor) -> Metric:
        raise NotImplementedError 
    
    def __add__(self, other: Metric) -> Metric:
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, getattr(self, attr) + getattr(other, attr))
        self.n += other.n
        return self 
    
    def __radd__(self, other: Metric) -> Metric:
        return self + other 
    
    def __repr__(self) -> str:
        return ', '.join(f'{metric}={getattr(self, metric)*(100 if metric in self.SCALE else 1):.2f}' for metric in self.METRICS)
        
    
    def improves(self, other: Metric) -> bool:
        if self.MODE == 'max':
            return any(getattr(self, metric) > getattr(other, metric) for metric in self.METRICS)
        else:
            return any(getattr(self, metric) < getattr(other, metric) for metric in self.METRICS)
    
    def __sub__(self, other: Metric) -> bool:
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, getattr(self, attr) - getattr(other, attr))
        return self 
        
        

class SegmentationMetric(Metric):
    METRICS = ['PACC', 'MACC', 'MIU', 'FWIU']
    ATTRIBUTES = ['pacc', 'macc', 'miu', 'fwiu']
    SCALE = ['PACC', 'MACC', 'MIU', 'FWIU']
    MODE = 'max'
    eps = 1e-12
    
    def __call__(self, preds: torch.Tensor, golds: torch.Tensor) -> SegmentationMetric:
        """Compute the semantic segmentation metrics.

        Args:
            preds (torch.Tensor): ``[batch_size, height, width]``.
            golds (torch.Tensor): ``[batch_size, height, width]``.

        Returns:
            SegmentationMetric.
        """
        labels, counts = golds.unique(return_counts=True)
        # true positives
        tps = torch.tensor([((golds == label) & (preds == label)).sum() for label in labels]).to(preds.device)
        # predicted positives
        pps = torch.tensor([(preds == label).sum() for label in labels]).to(preds.device)
        
        # pixel accuracy 
        self.pacc += (golds == preds).sum()/counts.sum()
        
        # mean accuracy
        self.macc += (1/len(labels))*(tps/counts).sum()
        
        # mean IU
        self.miu += (1/len(labels))*(tps/(counts+pps-tps)).sum()
        
        # frequency weighted IU
        self.fwiu += (counts*(tps/(counts+pps-tps))).sum()/counts.sum()
        
        self.n += 1
        
        return self 
        
    @property
    def PACC(self) -> float:
        return float(self.pacc)/(self.n+self.eps)
    
    @property
    def MACC(self) -> float:
        return float(self.macc)/(self.n +self.eps)
    
    @property
    def MIU(self) -> float:
        return float(self.miu)/(self.n +self.eps)
    
    @property 
    def FWIU(self) -> float:
        return self.fwiu/(self.n+self.eps)
    
    
class GenerationMetric(SegmentationMetric):
    METRICS = ['rPACC', 'rMACC', 'rMIU', 'rFWIU', 'fPACC', 'fMACC', 'fMIU', 'fFWIU']
    ATTRIBUTES = ['real_pacc', 'real_macc', 'real_miu', 'real_fwiu', 'fake_pacc', 'fake_macc', 'fake_miu', 'fake_fwiu']
    SCALE =  ['rPACC', 'rMACC', 'rMIU', 'rFWIU', 'fPACC', 'fMACC', 'fMIU', 'fFWIU']
    MODE = 'max'
    eps = 1e-12
    
    def __call__(self, real_segmented: torch.Tensor, fake_segemented: torch.Tensor, labels: torch.Tensor):
        real_metric = SegmentationMetric(real_segmented, labels)
        fake_metric = SegmentationMetric(fake_segemented, labels)
        for attr in real_metric.ATTRIBUTES:
            self.__setattr__(f'real_{attr}', getattr(self, f'real_{attr}') + getattr(real_metric, attr))
        for attr in fake_metric.ATTRIBUTES:
            self.__setattr__(f'fake_{attr}', getattr(self, f'fake_{attr}') + getattr(fake_metric, attr))
        self.n += 1 
        return self 
    
    @property
    def rPACC(self) -> float:
        return float(self.real_pacc)/(self.n+self.eps)
    
    @property
    def rMACC(self) -> float:
        return float(self.real_macc)/(self.n +self.eps)
    
    @property
    def rMIU(self) -> float:
        return float(self.real_miu)/(self.n +self.eps)
    
    @property 
    def rFWIU(self) -> float:
        return self.real_fwiu/(self.n+self.eps)
    
    @property
    def fPACC(self) -> float:
        return float(self.fake_pacc)/(self.n+self.eps)
    
    @property
    def fMACC(self) -> float:
        return float(self.fake_macc)/(self.n +self.eps)
    
    @property
    def fMIU(self) -> float:
        return float(self.fake_miu)/(self.n +self.eps)
    
    @property 
    def fFWIU(self) -> float:
        return self.fake_fwiu/(self.n+self.eps)
        