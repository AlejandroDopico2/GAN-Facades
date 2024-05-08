from models import SemanticSegmenter, AdversarialTranslator
from utils import FacadesDataset
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Image translation system')
    parser.add_argument('gen_type', type=str, help='Generator type')
    parser.add_argument('--path', '-p', type=str, help='Path where to store the image translator')
    parser.add_argument('--data', type=str, default='facades', help='Facades dataset folder')
    parser.add_argument('--segmenter', '-s', type=str, default='results/segmenter', help='Segmenter path')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='CUDA device')
    parser.add_argument('--weights', action='store_true', help='Whether to use weighted loss')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')

    args = parser.parse_args()
    data = FacadesDataset.from_folder(args.data)    
    train, dev, test = data.split(0.1, 0.1)
    
    segmenter = SemanticSegmenter.load(args.segmenter, device=args.device)
    adversarial = AdversarialTranslator.build(segmenter, gen_type=args.gen_type, weights=args.weights, device=args.device)
    adversarial.train(train, dev, test, path=args.path, lr=2e-4, batch_size=1, aug=True)