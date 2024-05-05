from models import SemanticSegmenter, AdversarialTranslator
from utils import FacadesDataset
from kmeans_pytorch import kmeans, kmeans_predict
import os
from torchvision.transforms import Compose, Lambda, ToTensor, RandomHorizontalFlip, RandomAffine

if __name__ == '__main__':
    data = FacadesDataset.from_folder('facades')
    train, dev = data.split(0.1)
    
    segmenter = SemanticSegmenter.build(device='cuda:1')
    segmenter.train(train, dev, path='results/segmenter/')
    # segmenter = SemanticSegmenter.load('results/segmenter', device='cuda:1')
    GENERATORS = ['base', 'deform', 'attn', 'link', 'fpn', 'psp', 'pan']
    for generator in GENERATORS:
        print(f'Executing {generator}')
        if not os.path.exists(f'results/{generator}/result.pickle'):
            # without data augmentation
            adversarial = AdversarialTranslator.build(segmenter, num_blocks=4, gen_type=generator, device='cuda:1')
            adversarial.train(train, dev, path=f'results/{generator}/', lr=2e-4, batch_size=1)
            os.system(f'rm results/{generator}/*.pt') # save space
    
        # with data augmentation 
        if not os.path.exists(f'results/{generator}-aug/result.pickle'):
            adversarial = AdversarialTranslator.build(segmenter, num_blocks=4, gen_type=generator, device='cuda:1')
            adversarial.train(train, dev, path=f'results/{generator}-aug/', lr=2e-4, batch_size=1, aug=True)
            os.system(f'rm results/{generator}-aug/*.pt') # save space
