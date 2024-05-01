import torch 

def denorm(img: torch.Tensor) -> torch.Tensor:
    """Denormalizes a normalized image (between -1 and 1)."""
    maxx, minx = img.max(), img.min()
    return (img+(maxx-minx)+minx)
    