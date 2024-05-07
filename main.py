from models import SemanticSegmenter, AdversarialTranslator
from utils import FacadesDataset

if __name__ == '__main__':
    data = FacadesDataset.from_folder('facades')
    train, dev, test = data.split(0.1, 0.1)
    
    # segmenter = SemanticSegmenter.build(n_labels=5, device='cuda:1')
    # segmenter.train(train, dev, test, path='results/segmenter/', aug=True)
    segmenter = SemanticSegmenter.load('results/segmenter2', device='cuda:1')
    adversarial = AdversarialTranslator.build(segmenter, gen_type='base', device='cuda:1')
    adversarial.train(train, dev, test, path=f'prueba-base/', lr=2e-4, batch_size=1, aug=True)
    