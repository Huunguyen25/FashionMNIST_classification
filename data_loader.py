"""Shared FashionMNIST data loading for the CNN and MLP."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# def dataloader(train_dataset, test_dataset, batch_size=64):
#     """Create shuffled training batches and ordered test batches."""
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader

def dataloader(train_dataset, test_dataset, batch_size=64):
    train_data, val_data = random_split(
        train_dataset,
        [54000, 6000],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def load_data(batch_size=64, root="./data/mnist", download=True):
    """Return image batches shaped (N, 1, 28, 28) and integer labels.
       The CNN uses the images directly; the MLP flattens them in its own loop.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=download, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=download, transform=transform
    )
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset, batch_size)
