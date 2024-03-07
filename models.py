import torch
import torch.nn as nn

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, 50)

        self.init_size = 7
        self.l1 = nn.Linear(latent_dim + 50, 128 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
        
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_size=28, n_classes=10):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, img_size * img_size)

        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),  # Change input channels to 2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),  # Adjusted input size
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(label_emb.size(0), 1, img.shape[2], img.shape[3])
        d_in = torch.cat((img, label_emb), 1)
        validity = self.model(d_in)
        return validity
