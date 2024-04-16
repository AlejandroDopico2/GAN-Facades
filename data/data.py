import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class FacadesDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path

        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jpg")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        real_image = image.crop((0, 0, 256, 256))
        input_image = image.crop((256, 0, 512, 256))

        if self.transform:
            real_image = self.transform(real_image)
            input_image = self.transform(input_image)

        return real_image, input_image
