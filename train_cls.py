#!/usr/bin/env python3
"""
Image Classifier Training Script using EfficientNet-B3
"""

import argparse
import os
import sys
import json
import secrets
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt

from common import EfficientNetClassifier, get_transforms


class ImageDataset(Dataset):
    """Custom dataset for image classification"""

    def __init__(self, positive_dir: str, negative_dir: str, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # Load positive images (label 1)
        positive_path = Path(positive_dir)
        if positive_path.exists():
            for img_path in positive_path.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(1)
            for img_path in positive_path.glob("*.jpeg"):
                self.images.append(str(img_path))
                self.labels.append(1)
            for img_path in positive_path.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(1)

        # Load negative images (label 0)
        negative_path = Path(negative_dir)
        if negative_path.exists():
            for img_path in negative_path.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(0)
            for img_path in negative_path.glob("*.jpeg"):
                self.images.append(str(img_path))
                self.labels.append(0)
            for img_path in negative_path.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(0)

        print(
            f"Loaded {len(self.images)} images: {sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, 300, 300)
            return dummy_image, label


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    acc: float,
    save_path: str,
):
    """Save model checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "acc": acc,
        },
        save_path,
    )


def plot_training_history(
    train_losses: List[float],
    train_accs: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: str,
):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B3 Image Classifier"
    )
    parser.add_argument(
        "--positive-dir",
        type=str,
        required=True,
        help="Directory containing positive images",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        required=True,
        help="Directory containing negative images",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=300, help="Input image size")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="log",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Generate experiment name
    experiment_name = f"cls_{secrets.token_hex(4)}"

    # Create log directory with experiment name
    log_dir = Path(args.save_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {experiment_name}")
    print(f"Log directory: {log_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup transforms
    train_transform, val_transform = get_transforms(args.image_size)

    # Create dataset
    dataset = ImageDataset(
        args.positive_dir, args.negative_dir, transform=train_transform
    )

    if len(dataset) == 0:
        print("No images found in the specified directories!")
        sys.exit(1)

    # Split dataset
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Update transforms for validation set
    val_dataset.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    model = EfficientNetClassifier(num_classes=2, pretrained=True)
    model = model.to(device)

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {checkpoint['epoch']}")
        else:
            print(f"Checkpoint file {args.resume} not found!")

    # Setup tensorboard with experiment name
    writer = SummaryWriter(log_dir / "tensorboard", comment=f"_{experiment_name}")

    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 20)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Log metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, log_dir / "best_model.pth"
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                val_acc,
                log_dir / f"checkpoint_epoch_{epoch + 1}.pth",
            )

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.epochs - 1,
        val_loss,
        val_acc,
        log_dir / "final_model.pth",
    )

    # Plot training history
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs, log_dir / "training_history.png"
    )

    # Save training config
    config = {
        "positive_dir": args.positive_dir,
        "negative_dir": args.negative_dir,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "image_size": args.image_size,
        "val_split": args.val_split,
        "final_train_loss": train_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_loss": val_losses[-1],
        "final_val_acc": val_accs[-1],
        "best_val_acc": best_val_acc,
    }

    with open(log_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"Results saved in: {log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
