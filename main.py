import torch
import matplotlib.pyplot as plt

import argparse

from pathlib import Path

from torch.utils.data import DataLoader, sampler
from torch.optim import Adam

from tqdm import tqdm

from torchvision import transforms as T
from utils.data import FacadesDataset
from utils.loss import DiscriminatorLoss, GeneratorLoss
from modules.unet import UNet
from modules.patch_gan import Discriminator

import numpy as np


def generate_images(model, test_input, tar, epoch, device):
    test_input_to_model = test_input[None, :, :, :]
    prediction = model(test_input_to_model.to(device)).detach().cpu().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    plt.figure(figsize=(15, 15))

    test_input = np.transpose(test_input, (1, 2, 0))

    display_list = [test_input, tar[0], prediction[0]]
    title = ["Ground Truth", "Input Image", "Predicted Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.savefig(f"images/generated_image{epoch}.png")
    plt.close()


def main(data_path: str = "facades", epochs: int = 100, batch_size: int = 1):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transforms = T.Compose(
        [
            T.ToTensor(),  # Convert PIL image to tensor
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            T.RandomHorizontalFlip(p=0.5),
        ]
    )

    dataset = FacadesDataset(data_path, transform=transforms)

    val_split = 0.1
    test_split = 0.1
    dataset_size = len(dataset)

    indices = list(range(dataset_size))

    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(np.floor((val_split + test_split) * dataset_size))
    train_indices, test_val_indices = indices[split:], indices[:split]

    split = int(np.floor((0.5) * len(test_val_indices)))
    val_indices, test_indices = test_val_indices[split:], test_val_indices[:split]

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)
    test_sampler = sampler.SubsetRandomSampler(test_indices)

    example_input, example_target = dataset[0]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler
    )  # TODO: use

    generator = UNet(num_blocks=8, filter_size=4).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    loss_G = GeneratorLoss(alpha=100)
    loss_D = DiscriminatorLoss()

    # no_improvement_count = 0
    # patience = 5

    Path("images/").mkdir(exist_ok=True, parents=True)

    best_val_loss = np.inf
    for epoch in range(epochs):
        ge_loss = 0.0
        de_loss = 0.0

        with tqdm(total=len(train_loader), desc="train") as bar:
            for real, x in train_loader:
                x = x.to(device)
                real = real.to(device)

                # Generator Loss
                fake = generator(x)
                fake_pred = discriminator(fake, x)
                generator_loss = loss_G(fake, real, fake_pred)

                # Discriminator Loss
                fake = generator(x).detach()
                fake_pred = discriminator(fake, x)
                real_pred = discriminator(real, x)
                discriminator_loss = loss_D(fake_pred, real_pred)

                # Generator Params Update
                optimizer_G.zero_grad()
                generator_loss.backward()
                optimizer_G.step()

                # Discriminator Params Update
                optimizer_D.zero_grad()
                discriminator_loss.backward()
                optimizer_D.step()

                ge_loss += generator_loss.item()
                de_loss += discriminator_loss.item()

                bar.update(1)

                bar.set_postfix(
                    {
                        "generator_loss": f"{ge_loss:.2f}",
                        "discriminator_loss": f"{de_loss:.2f}",
                    }
                )

        # Validation loop
        with torch.no_grad():
            val_loss = 0.0
            for val_real, val_x in val_loader:
                val_x = val_x.to(device)
                val_real = val_real.to(device)

                val_fake = generator(val_x)
                val_fake_pred = discriminator(val_fake, val_x)
                val_loss += loss_G(val_fake, val_real, val_fake_pred).item()

            val_loss /= len(val_loader)

        # Check for Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            # Save models when val loss improves (decreases)
            torch.save(generator.state_dict(), "models/generator.pt")
            torch.save(discriminator.state_dict(), "models/discriminator.pt")
            print("Saved models at epoch", epoch + 1)
        else:
            no_improvement_count += 1

        samples = len(train_loader.dataset)
        ge_loss /= samples
        de_loss /= samples
        print(
            f"Epoch {epoch + 1}: generator_loss: {ge_loss:.2f}, discriminator_loss: {de_loss:.2f}, val_loss: {val_loss:.2f}"
        )

        # Check for Early Stopping
        # if no_improvement_count >= patience:
        #     print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
        #     break

        generate_images(generator, example_input, example_target, epoch + 1, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="facades")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)
