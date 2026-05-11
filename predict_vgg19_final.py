"""
Prediction script for Songket motif inference using VGG19.
Loads 'songket_motif_vgg19_final.pth' and predicts motif for test images.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update to match your dataset classes (alphabetical order)
class_names = ['bunga_pecah_lapan', 'pucuk_rebung', 'tampuk_manggis']

# Preprocess (match training normalization)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_trained_model(model_path: str):
    """Load VGG19 model weights."""
    model = models.vgg19_bn()
    # Replace the final classifier layer
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_motif(image_path: str, model: torch.nn.Module):
    if not os.path.exists(image_path):
        print(f"❌ Error: Cannot find {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    batch = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, 0)

    print("\n" + "=" * 30)
    print(f"📸 Image: {image_path}")
    print(f"🎯 Result: {class_names[index].upper()}")
    print(f"📈 Confidence: {confidence.item() * 100:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    MODEL_FILE = "songket_motif_vgg19_final.pth"
    TEST_IMAGE = "test_image.jpg"

    if os.path.exists(MODEL_FILE):
        print(f"🧠 Loading model from {MODEL_FILE}...")
        my_model = load_trained_model(MODEL_FILE)
        predict_motif(TEST_IMAGE, my_model)
    else:
        print(f"❌ Error: {MODEL_FILE} not found. Did you run train_vgg19_final.py?")
