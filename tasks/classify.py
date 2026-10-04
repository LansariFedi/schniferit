import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_train_transform(img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def default_val_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_loader(data_dir, split, transform, batch_size=32, workers=4):
    ds = datasets.ImageFolder(f"{data_dir}/{split}", transform=transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=workers
    )


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += yb.size(0)
    return correct / total
