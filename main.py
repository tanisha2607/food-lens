import os
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torch
import torchvision.transforms as transforms

import config
from model import create_model

FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyro", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

def get_image():
    root = tk.Tk()
    root.withdraw()

    print("Opening file dialog window... Please pick an image.")

    file_path = filedialog.askopenfilename(
        title="Select a Food Photo for AI Inference",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.webp"),
            ("All Files", "*.*")
        ]
    )
    return file_path if file_path != "" else None

def main():
    print("\nFOOD-101 PREDICTION TERMINAL")

    img_path = get_image()
    if not img_path:
        print("[EXIT] No file selected. Closing script.")
        return
    print(f"[INFO] Selected Image: {os.path.basename(img_path)}")

    model = create_model()
    checkpoint_path = f"{config.CHECKPOINT_DIR}/food101_epoch_10.pth"

    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Missing checkpoint! Place 'food101_epoch_10.pth' inside a '{config.CHECKPOINT_DIR}' folder.")
        return

    print("[INFO] Loading 76.94% accuracy fine-tuned weights...")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(config.DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        raw_image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(raw_image)
        input_batch = input_tensor.unsqueeze(0).to(config.DEVICE)
    except Exception as e:
        print(f"[ERROR] Failed to process image file. Details: {e}")
        return
    
    print("[INFO] Crunching calculations...")
    with torch.no_grad():
        logits = model(input_batch)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    food_name = FOOD101_CLASSES[predicted_idx.item()].replace("_", " ").title()

    print("\n=================== AI PREDICTION RESULT ===================")
    print(f"🎯 Predicted Category : {food_name}")
    print(f"📊 Confidence Score    : {confidence.item() * 100:.2f}%")
    print("============================================================\n")


if __name__ == "__main__":
    main()