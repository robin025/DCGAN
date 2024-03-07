from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataloader = DataLoader(
        FashionMNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader
