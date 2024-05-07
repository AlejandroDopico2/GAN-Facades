from models import SemanticSegmenter, AdversarialTranslator
from utils import FacadesDataset
from kmeans_pytorch import kmeans, kmeans_predict
import os
from torchvision.transforms import Compose, Lambda, ToTensor, RandomHorizontalFlip, RandomAffine

if __name__ == '__main__':
    data = FacadesDataset.from_folder('facades')
    train, dev, test = data.split(0.1, 0.1)
    
    # segmenter = SemanticSegmenter.build(device='cuda:1')
    # segmenter.train(train, dev, test, path='results/segmenter/')
    segmenter = SemanticSegmenter.load('results/segmenter', 'cuda:1')
    adversarial = AdversarialTranslator.build(segmenter, gen_type='attn', device='cuda:1')
    adversarial.train(train, dev, test, path=f'results/attn/', lr=2e-4, batch_size=1, aug=True)
    
