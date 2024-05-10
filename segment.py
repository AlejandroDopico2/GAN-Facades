from models import SemanticSegmenter
from utils import FacadesDataset
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="Image translation system")
    parser.add_argument("-n", type=int, default=5, help="Number of labels")
    parser.add_argument(
        "--path", "-p", type=str, help="Path where to store the image translator"
    )
    parser.add_argument(
        "--data", type=str, default="facades", help="Facades dataset folder"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda:0", help="CUDA device"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size.")

    args = parser.parse_args()
    data = FacadesDataset.from_folder(args.data)
    train, dev, test = data.split(0.1, 0.1)

    segmenter = SemanticSegmenter.build(args.n, device=args.device)
    segmenter.train(
        train, dev, test, path=args.path, lr=1e-4, batch_size=args.batch_size, aug=True
    )
