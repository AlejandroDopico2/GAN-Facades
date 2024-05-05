from __future__  import annotations
import os, torch, random 
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Union
import numpy as np 
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine


class FacadesDataset(Dataset):
    IMG_SIZE = (256, 256)
    
    def __init__(self, image_paths: List[str], transform: Optional[Compose] = None):
        self.image_paths = image_paths
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __iter__(self):
        return iter(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        real_image = image.crop((0, 0, 256, 256))
        input_image = image.crop((256, 0, 512, 256))
        
        if self.transform:
            # set random seed 
            seed = np.random.randint(123)
            torch.manual_seed(seed)
            real_image = self.transform(real_image)
            torch.manual_seed(seed)
            input_image = self.transform(input_image)
        
        return real_image, input_image
    
    def split(self, *sizes: List[Union[int, float]], shuffle: bool = True) -> Tuple[FacadesDataset]:
        if shuffle:
            random.shuffle(self.image_paths)
        lens = [int(len(self)*ratio) for ratio in sizes] if isinstance(sizes[0], float) else sizes
        indices = [0] + np.cumsum(lens).tolist()
        splits = [FacadesDataset(self.image_paths[indices[i]:indices[i+1]], self.transform) for i in range(len(indices)-1)]
        return FacadesDataset(self.image_paths[indices[-1]:], self.transform), *splits
    
    @classmethod
    def from_folder(cls, folder_path: str, transform: Optional[Compose] = None):
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jpg")
        ]
        return cls(image_paths, transform)