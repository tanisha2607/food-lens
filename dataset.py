import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import constants from your config file
import config


def get_data_loaders():
    """Creates and returns the train and test DataLoaders for Food-101."""

    # 1. ImageNet mean and std values for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # 2. Train transform pipeline
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    # 3. Test transform pipeline
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    # 4. Load Food-101 train and test splits pointing to config.DATA_DIR
    print("Loading Food-101 dataset (this may take a moment)...")

    train_dataset = torchvision.datasets.Food101(
        root=config.DATA_DIR,
        split="train",
        transform=train_transform,
        download=True,  # Automatically downloads if not found in DATA_DIR
    )

    test_dataset = torchvision.datasets.Food101(
        root=config.DATA_DIR, split="test", transform=test_transform, download=True
    )

    # 5. Two DataLoaders using config.BATCH_SIZE
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Train loader shuffles
        num_workers=4,  # Speeds up data loading (optional)
        pin_memory=True,  # Speeds up data transfer to GPU (optional)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # Test loader does not shuffle
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


# Quick test snippet to verify everything loads correctly
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    print("\n--- Data Loaders Ready ---")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Peek at one batch
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  
    print(f"Batch Label Shape: {labels.shape}")  