import torch
import numpy as np
from modules.unet import UNet
from modules.patch_gan import Discriminator

from random import randint

import os
from PIL import Image
import matplotlib.pyplot as plt


def generate_images(model, test_input, tar, i, device):
    test_to_model = torch.permute(test_input, (0, 3, 1, 2)).to(device)
    prediction = model(test_to_model).detach().cpu().numpy()

    prediction = np.transpose(prediction, (0, 2, 3, 1))

    print("test_input", test_input.shape)
    print("prediction", prediction.shape)
    print("tar", tar.shape)

    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        img = display_list[i]
        plt.imshow(img * 0.5 + 0.5)
        plt.axis("off")
    plt.savefig("images/predicted_image.png")


paths = [
    os.path.join("./facades/", f)
    for f in os.listdir("./facades/")
    if f.endswith(".jpg")
]

idx = randint(0, len(paths) - 1)

image_path = paths[idx]

img = Image.open(image_path)

left_half = img.crop((0, 0, 256, 256))
right_half = img.crop((256, 0, 512, 256))

left_half = np.array(left_half) / 255
right_half = np.array(right_half) / 255

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet(num_blocks=8, filter_size=4)
discriminator = Discriminator()

unet.load_state_dict(torch.load("models/generator.pt"))
discriminator.load_state_dict(torch.load("models/discriminator.pt"))

unet = unet.to(device)
discriminator = discriminator.to(device)

generate_images(
    unet,
    torch.tensor(left_half, dtype=torch.float32).unsqueeze(0),
    torch.tensor(right_half, dtype=torch.float32).unsqueeze(0),
    0,
    device,
)
