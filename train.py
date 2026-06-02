import torch
import torch.nn as nn
import torch.optim as optim
import os

from dataset import get_data_loaders
from model import create_model
import config

def train():
    model = create_model()
    train_loader, test_loader = get_data_loaders()

    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_parameters, lr=config.LEARNING_RATE)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok = True)
    print(f"\nStarting training on device: {config.DEVICE.upper()}...")

    for epoch in range(1,config.NUM_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predictions = torch.max(outputs, 1)
            running_corrects += torch.sum(predictions == labels.data).item()
            total_samples += images.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch}/{config.NUM_EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}]"
                )
        
        epoch_loss = running_loss/total_samples
        epoch_acc = (running_corrects/total_samples)*100

        print(f"\n=== Epoch {epoch} Complete ===")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Train Accuracy: {epoch_acc:.2f}%\n")

        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"food101_mobilenetv2_epoch_{epoch}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to: {checkpoint_path}\n")

if __name__ == "__main__":
    train()
