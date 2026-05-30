import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import config


def get_data_loaders():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    print("Loading Food-101 dataset (this may take a moment)...")

    train_dataset = torchvision.datasets.Food101(
        root=config.DATA_DIR,
        split="train",
        transform=train_transform,
        download=True,
    )

    test_dataset = torchvision.datasets.Food101(
        root=config.DATA_DIR, split="test", transform=test_transform, download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Train loader shuffles
        num_workers=0,  # Speeds up data loading (optional)
        pin_memory=True,  # Speeds up data transfer to GPU (optional)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    print("\n--- Data Loaders Ready ---")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  
    print(f"Batch Label Shape: {labels.shape}")  