import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from dataset import get_dataloader
from models import Generator, Discriminator
from train import train
from utils import weights_init
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import os

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = Generator(latent_dim, n_classes).to(device)
    discriminator = Discriminator(img_size, n_classes).to(device)

    # Apply the weights_init function to randomly initialize all weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Configure data loader
    dataloader = get_dataloader(img_size, batch_size)

    # Loss function
    adversarial_loss = nn.MSELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Training
    losses = train(device, dataloader, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, n_epochs, latent_dim, n_classes)

    # Plot and save Losses
    plt.figure(figsize=(10,5))
    plt.plot(losses["G"], label="Generator Loss")
    plt.plot(losses["D"], label="Discriminator Loss")
    plt.title("Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(results_dir, 'loss_curves.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # Generate after training and save images
    n_samples = 100
    z = torch.randn(n_samples, latent_dim, device=device)
    labels = torch.randint(0, n_classes, (n_samples,), device=device)
    with torch.no_grad():
        gen_imgs = generator(z, labels).cpu()
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(gen_imgs, padding=2, normalize=True), (1, 2, 0)))
    generated_img_path = os.path.join(results_dir, 'generated_images.png')
    plt.savefig(generated_img_path)
    plt.close()
