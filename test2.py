from models import SemanticSegmenter, AdversarialTranslator
from utils import FacadesDataset
from kmeans_pytorch import kmeans, kmeans_predict
import os, time
from torchvision.transforms import Compose, Lambda, ToTensor, RandomHorizontalFlip, RandomAffine

if __name__ == '__main__':
    data = FacadesDataset.from_folder('facades')
    train, dev, test = data.split(0.1, 0.1)
    
    
    while not os.path.exists('results/segmenter/result.pickle'):
        print('Waiting for the segmenter')
        time.sleep(60*10)
    segmenter = SemanticSegmenter.load('results/segmenter/', device='cuda:1')
    GENERATORS = ['link', 'fpn', 'psp']
    for generator in GENERATORS:
        print(f'Executing {generator}')
        if not os.path.exists(f'results/{generator}/result.pickle'):
            adversarial = AdversarialTranslator.build(segmenter, gen_type=generator, device='cuda:1')
            adversarial.train(train, dev, test, path=f'results/{generator}/', lr=2e-4, batch_size=1, aug=True)
            os.system(f'rm results/{generator}/*.pt') # save space
    
