# 🍜 Food Lens

> A deep learning image classifier that identifies **101 food categories** from a single photo — built from scratch using transfer learning on MobileNet V2.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-76.94%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Food--101-orange?style=flat-square)

---

## What it does

Point it at any food photo and FoodLens will tell you what it is — from apple pie to waffles, across 101 categories. No APIs. No cloud calls. Everything runs locally on your machine.

```
$ python main.py

FOOD-101 PREDICTION TERMINAL
Opening file dialog window... Please pick an image.

=================== AI PREDICTION RESULT ===================
🎯 Predicted Category : Sushi
📊 Confidence Score    : 89.43%
============================================================
```

---

## How it works

FoodLens uses **transfer learning** on top of MobileNet V2 — a lightweight convolutional neural network pretrained on ImageNet. Rather than training from scratch (which would require weeks of compute), we:

1. Load MobileNet V2 with its pretrained ImageNet weights
2. Freeze the base feature extraction layers
3. Unfreeze the last few blocks for fine-tuning
4. Replace the final classification head with a new layer outputting 101 classes
5. Train on the Food-101 dataset using the Adam optimizer

This approach gives strong accuracy with far less compute than training from scratch.

---

## Results

| Metric | Value |
|---|---|
| Dataset | Food-101 (101,000 images, 101 classes) |
| Model | MobileNet V2 (fine-tuned) |
| Training epochs | 10 |
| Final train accuracy | 64.11% |
| **Final test accuracy** | **76.94%** |

The test accuracy exceeding train accuracy is expected — training uses heavy augmentation (random crops, flips) which makes it harder, while testing uses clean center crops.

---

## Project Structure

```
food-lens/
│
├── config.py          # Shared constants (batch size, lr, device, paths)
├── dataset.py         # Food-101 DataLoaders with train/test transforms
├── model.py           # MobileNet V2 with custom classification head
├── train.py           # Training loop with checkpointing
├── evaluate.py        # Final test set evaluation
├── main.py            # Inference — pick an image, get a prediction
│
├── data/              # Food-101 dataset (auto-downloaded)
└── checkpoints/       # Saved model weights per epoch
```

---

## Setup & Usage

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/food-lens.git
cd food-lens
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install torch torchvision pillow
```

**3. Train the model** *(recommended on Google Colab with GPU)*
```bash
python train.py
```
This will auto-download the Food-101 dataset (~5GB) on first run and save checkpoints after each epoch.

**4. Evaluate**
```bash
python evaluate.py
```

**5. Run inference**
```bash
python main.py
```
A file dialog will open — pick any food photo and the model will predict what it is.

---

## Training on Google Colab

For faster training, use Google Colab with a free T4 GPU:

1. Push your code to GitHub
2. Clone the repo in a Colab notebook
3. Set runtime to GPU — `Runtime → Change Runtime Type → T4 GPU`
4. Run `train.py`
5. Download the checkpoint from `checkpoints/` before closing the session

Training 10 epochs on a T4 GPU takes roughly 1-2 hours.

---

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 101,000 images across 101 food categories, with 750 training images and 250 test images per class. Sourced automatically via `torchvision.datasets.Food101`.

---

## Built by

[Tanisha](https://github.com/tanisha2607) & [Shreyans](https://github.com/shreyans-gits)
