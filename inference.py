#!/usr/bin/env python3
"""
Inference script for trained EfficientNet-B3 classifier
"""

import argparse
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

from common import EfficientNetClassifier, get_transforms


def download_model_from_hf(repo_id, filename):
    """Download model from Hugging Face Hub"""
    print(f"Downloading model from Hugging Face: {repo_id}/{filename}")
    model_path = hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir="./models"
    )
    print(f"Model downloaded to: {model_path}")
    return model_path


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint"""
    model = EfficientNetClassifier(num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_image(model: nn.Module, image_path: str, transform, device: torch.device):
    """Predict single image"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence, probabilities[0].cpu().numpy()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Inference with trained EfficientNet-B3 classifier"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model checkpoint (optional, will download from HF if not provided)",
    )
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument(
        "--image-dir", type=str, help="Directory containing images for batch prediction"
    )
    parser.add_argument("--image-size", type=int, default=300, help="Input image size")
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.txt",
        help="Output file for batch predictions",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine model path
    if args.model:
        model_path = args.model
        print(f"Using local model: {model_path}")
    else:
        # Download from Hugging Face
        model_path = download_model_from_hf(
            "Curioushi61/BoxAutoLabel", "box_classification.pth"
        )

    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path, device)

    # Setup transform
    _, transform = get_transforms(args.image_size)

    if args.image:
        # Single image prediction
        predicted_class, confidence, probabilities = predict_image(
            model, args.image, transform, device
        )

        if predicted_class is not None:
            class_name = "Positive" if predicted_class == 1 else "Negative"
            print(f"Image: {args.image}")
            print(f"Prediction: {class_name} (class {predicted_class})")
            print(f"Confidence: {confidence:.4f}")
            print(
                f"Probabilities: [Negative: {probabilities[0]:.4f}, Positive: {probabilities[1]:.4f}]"
            )

    elif args.image_dir:
        # Batch prediction
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Image directory {args.image_dir} does not exist!")
            return

        results = []
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))

        for image_path in tqdm(image_paths, desc="Predicting images"):
            predicted_class, confidence, probabilities = predict_image(
                model, str(image_path), transform, device
            )

            if predicted_class is not None:
                class_name = "Positive" if predicted_class == 1 else "Negative"
                result = {
                    "image": str(image_path),
                    "prediction": class_name,
                    "class_id": predicted_class,
                    "confidence": confidence,
                    "prob_negative": probabilities[0],
                    "prob_positive": probabilities[1],
                }
                results.append(result)

        results = sorted(results, key=lambda x: x["image"])

        # Save results to file
        with open(args.output, "w") as f:
            f.write(
                "Image,Prediction,Class_ID,Confidence,Prob_Negative,Prob_Positive\n"
            )
            for result in results:
                f.write(
                    f"{result['image']},{result['prediction']},{result['class_id']},"
                    f"{result['confidence']:.4f},{result['prob_negative']:.4f},{result['prob_positive']:.4f}\n"
                )

        print(f"\nProcessed {len(results)} images")
        print(f"Results saved to {args.output}")

    else:
        print(
            "Please specify either --image for single prediction or --image-dir for batch prediction"
        )


if __name__ == "__main__":
    main()
