import torch
import os

BATCH_SIZE = 32
NUM_CLASSES = 101
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = os.path.join(os.getcwd(), "data")
CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")