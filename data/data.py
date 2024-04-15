import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FacadesDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path

        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        left_half = image.crop((0, 0, 256, 256))
        right_half = image.crop((256, 0, 512, 256))

        if self.transform:
            left_half = self.transform(left_half)
            right_half = self.transform(right_half)

        return left_half, right_half

import matplotlib.pyplot as plt

# Example usage:
if __name__ == "__main__":
    # Define the path to your dataset folder
    folder_path = "./facades/"

    # Define any image transformations (optional)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        # Add more transforms here if needed
    ])

    # Create an instance of the custom dataset
    dataset = FacadesDataset(folder_path, transform=transform)

    # Example: Load the first sample from the dataset
    sample_idx = 0
    left_half, right_half = dataset[sample_idx]

    # Display the shapes of the image halves
    print("Left Half Shape:", left_half.shape)    # Should be (channels, height, width)
    print("Right Half Shape:", right_half.shape)  # Should be (channels, height, width)

    plt.figure()
    plt.imshow(torch.permute(left_half, (1, 2, 0)))
    plt.savefig('test.jpg')
    # plt.figure()
    # plt.imshow(right_half)
