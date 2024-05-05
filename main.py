from models import SemanticSegmenter, AdversarialTranslator
from utils import FacadesDataset

if __name__ == '__main__':
    data = FacadesDataset.from_folder('facades')
    train, dev = data.split(0.2)
    
    # segmenter = SemanticSegmenter.build(n_labels=5, device='cuda:0')
    # segmenter.train(train, dev, path='results/segmenter/', aug=True)
    segmenter = SemanticSegmenter.load('results/segmenter', device='cuda:0')
    adversarial = AdversarialTranslator.build(segmenter, gen_type='base', device='cuda:0')
    adversarial.train(train, dev, path=f'prueba/', lr=2e-4, batch_size=1, aug=True)