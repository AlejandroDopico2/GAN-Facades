from models import SemanticSegmenter, AdversarialTranslator
from utils import FacadesDataset
from kmeans_pytorch import kmeans, kmeans_predict
import torch 
from torchvision.transforms import Compose, Lambda, ToTensor, RandomHorizontalFlip, RandomAffine

if __name__ == '__main__':
    data = FacadesDataset.from_folder('facades')
    train, dev = data.split(0.1)
    
    # segmenter = SemanticSegmenter.build(device='cuda:1')
    # segmenter.train(train, dev, path='results/segmenter/')
    segmenter = SemanticSegmenter.load('results/segmenter/', device='cuda:1')
    
    adversarial = AdversarialTranslator.build(segmenter, device='cuda:1')
    adversarial.train(train, dev, path='results/adversarial/')
    adversarial = AdversarialTranslator.load('results/adversarial/', 'cuda:1')
    adversarial.predict(dev, 'results/adversarial/', 10)
    