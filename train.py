import torch
import torch.nn as nn
import torch.optim as optim
from utils import weights_init, label_smoothing
from models import Generator, Discriminator
from config import *
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def train(device, dataloader, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, n_epochs, latent_dim, n_classes):
    losses = {"G": [], "D": []}

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            smoothed_label_value = 0.75
            fake_label_value = 0.25
            valid = torch.full((imgs.size(0), 1), smoothed_label_value, device=device)
            fake = torch.full((imgs.size(0), 1), fake_label_value, device=device)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_labels = torch.randint(0, n_classes, (imgs.size(0),), device=device)

            gen_imgs = generator(z, gen_labels)

            optimizer_D.zero_grad()

            real_pred = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid)

            fake_pred = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)

            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            losses["G"].append(g_loss.item())
            losses["D"].append(d_loss.item())

            if i % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %.3f] [G loss: %.3f]"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

    return losses
