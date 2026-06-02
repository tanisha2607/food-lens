import torch
import torch.nn as nn
import torchvision.models as models

import config


def create_model():
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    for block in model.features[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2), nn.Linear(in_features=in_features, out_features=config.NUM_CLASSES)
    )

    model = model.to(config.DEVICE)
    return model


if __name__ == "__main__":
    my_food_model = create_model()

    print("--- Model Verification ---")

    early_param = next(my_food_model.features[0].parameters())
    print(f"Early feature layer frozen? {not early_param.requires_grad}")

    late_param = next(my_food_model.features[18].parameters())
    print(f"Deep feature layers unfrozen/trainable? {late_param.requires_grad}")

    classifier_param = my_food_model.classifier[1].weight
    print(f"New classifier layer trainable? {classifier_param.requires_grad}")

    print(f"New classifier architecture:\n{my_food_model.classifier}")

    dummy_input = torch.randn(1, 3, 224, 224).to(config.DEVICE)
    output = my_food_model(dummy_input)
    print(f"Output shape (expected [1, 101]): {list(output.shape)}")