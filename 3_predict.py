import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. SETUP: Device and Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE = "songket_motif_model.pth"
DATA_DIR = "dataset/train_tiles"  # Use this to get class names automatically

def get_class_names(path):
    """Automatically gets the list of motifs from your folder names."""
    if os.path.exists(path):
        # Alphabetical order is standard for PyTorch ImageFolder
        return sorted(os.listdir(path))
    else:
        # Fallback if folders are missing
        return ['bunga_pecah_lapan', 'pucuk_rebung', 'tampuk_manggis']

class_names = get_class_names(DATA_DIR)

def load_trained_model(model_path):
    """Rebuilds the ResNet50 architecture and loads your saved weights."""
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    # Dynamically set the number of classes based on your folders
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 2. PREPROCESS (No changes needed here)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_motif(image_path, model):
    if not os.path.exists(image_path):
        print(f"❌ Error: Cannot find {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    batch = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, 0)

    print("\n" + "="*40)
    print(f"📸 Image: {image_path}")
    print(f"🎯 Result: {class_names[index].upper().replace('_', ' ')}")
    print(f"📈 Confidence: {confidence.item()*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    TEST_IMAGE = "test_image.jpg"

    if os.path.exists(MODEL_FILE):
        print(f"🧠 Found {len(class_names)} motifs: {class_names}")
        print(f"🚀 Loading weights from {MODEL_FILE}...")
        my_model = load_trained_model(MODEL_FILE)
        predict_motif(TEST_IMAGE, my_model)
    else:
        print(f"❌ Error: {MODEL_FILE} not found. Run 2_train.py first!")