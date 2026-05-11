"""
Prediction script for Songket motif inference using GoogLeNet.
Loads 'songket_motif_googlenet_final.pth' and predicts motif for test images.
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
    """Load GoogLeNet weights, auto-detecting whether aux heads were saved."""
    state = torch.load(model_path, map_location=device)
    # Detect presence of aux heads in checkpoint keys
    has_aux = any((k.startswith("aux1") or k.startswith("aux2")) for k in state.keys())

    model = models.googlenet(aux_logits=has_aux, init_weights=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    if has_aux and getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
        model.aux1.fc = nn.Linear(model.aux1.fc.in_features, len(class_names))
    if has_aux and getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
        model.aux2.fc = nn.Linear(model.aux2.fc.in_features, len(class_names))

    # Load with strict=False to tolerate minor head-shape differences
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"⚠️  Missing keys ignored: {missing}")
    if unexpected:
        print(f"⚠️  Unexpected keys ignored: {unexpected}")

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
    MODEL_FILE = "songket_motif_googlenet_final.pth"
    TEST_IMAGE = "test_image.jpg"

    if os.path.exists(MODEL_FILE):
        print(f"🧠 Loading model from {MODEL_FILE}...")
        my_model = load_trained_model(MODEL_FILE)
        predict_motif(TEST_IMAGE, my_model)
    else:
        print(f"❌ Error: {MODEL_FILE} not found. Did you run train_googlenet_final.py?")
