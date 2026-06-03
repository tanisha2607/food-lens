import torch
import torch.nn as nn
import os

import config
from dataset import get_data_loaders
from model import create_model

def evaluate():
    print("Initializing evaluation script...")

    model = create_model()

    checkpoint_path = f"{config.CHECKPOINT_DIR}/food101_epoch_10.pth"
    
    print(f"Looking for weights at: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Missing checkpoint! Please upload your 'food101_epoch_10.pth' file into the '{config.CHECKPOINT_DIR}' folder in the sidebar.")
        return

    print("Loading weights into model skeleton...")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(config.DEVICE)

    model.eval()

    _, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    print(f"\nRunning final evaluation on {len(test_loader.dataset)} test images...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predictions = torch.max(outputs, 1)  # Get index of the highest score
            running_corrects += torch.sum(predictions == labels.data).item()
            total_samples += images.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"Evaluated Batch [{batch_idx + 1}/{len(test_loader)}]")

    final_loss = running_loss / total_samples
    final_accuracy = (running_corrects / total_samples) * 100

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Test Images Sampled : {total_samples}")
    print(f"Final Test Loss            : {final_loss:.4f}")
    print(f"Final Top-1 Test Accuracy   : {final_accuracy:.2f}%")
    print("====================================================\n")

if __name__ == "__main__":
    evaluate()