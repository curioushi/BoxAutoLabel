#!/usr/bin/env python3
"""
Common utilities for image classification training and inference
"""

import torch.nn as nn
import torchvision.transforms as transforms
import timm


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B3 based classifier"""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(EfficientNetClassifier, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b3", pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


def get_transforms(image_size: int = 300):
    """Get training and validation transforms"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform
