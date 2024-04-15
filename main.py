import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from torchvision import transforms as T
from data.data import FacadesDataset
from utils.loss import DiscriminatorLoss, GeneratorLoss
from models.unet import UNet
from models.patch_gan import Discriminator

import numpy as np

def generate_images(model, test_input, tar, epoch, device):
    test_input_to_model = test_input[None, :, :, :]
    prediction = model(test_input_to_model.to(device)).detach().cpu().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    plt.figure(figsize=(15,15))

    test_input = np.transpose(test_input, (1, 2, 0))

    display_list = [test_input, tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(f"images/generated_image{epoch}.png")
    plt.close()

def main(data_path:str = "facades"):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    transforms = T.Compose([
        T.ToTensor(),  # Convert PIL image to tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset = FacadesDataset(data_path, transform=transforms)

    example_input, example_target = dataset[0]
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    generator = UNet(num_blocks=8, filter_size=4).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    loss_G = GeneratorLoss(alpha=100)
    loss_D = DiscriminatorLoss()

    for epoch in range(50):
        ge_loss = 0.
        de_loss = 0.

        best_loss = 1000.
        with tqdm(total=len(data_loader), desc='train') as bar:
            for real, x in data_loader:
                x = x.to(device)
                real = real.to(device)

                fake = generator(x)

                fake_pred = discriminator(fake, x)
                generator_loss = loss_G(fake, real, fake_pred)

                fake = generator(x).detach()
                fake_pred = discriminator(fake, x)
                real_pred = discriminator(real, x)
                discriminator_loss = loss_D(fake_pred, real_pred)

                optimizer_G.zero_grad()
                generator_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                discriminator_loss.backward()
                optimizer_D.step()

                ge_loss += generator_loss.item()
                de_loss += discriminator_loss.item()

                bar.update(1)

                bar.set_postfix({'generator_loss': f'{ge_loss:.2f}', 'discriminator_loss': f'{de_loss:.2f}'})

        if ge_loss < best_loss:
            best_loss = ge_loss
            torch.save(generator.state_dict(), "models/generator.pt")
            torch.save(discriminator.state_dict(), "models/discriminator.pt")
        samples = len(dataset)
        ge_loss = ge_loss / samples
        de_loss = de_loss / samples
        print(f'Epoch {epoch+1}: generator_loss: {ge_loss:.2f}, discriminator_loss: {de_loss:.2f}')

        generate_images(generator, example_input, example_target, epoch+1, device)


if __name__ == '__main__':
    main()