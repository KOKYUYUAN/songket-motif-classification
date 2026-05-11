"""
Evaluation script to generate confusion matrix for the trained model.
Run after training to visualize classification performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models
import torch.nn as nn
import os

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['bunga_pecah_lapan', 'pucuk_rebung', 'tampuk_manggis']
data_dir = "dataset"
val_dir = os.path.join(data_dir, "val_tiles")

# Data transforms (same as validation in training)
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load validation dataset
val_dataset = datasets.ImageFolder(val_dir, val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load trained model
def load_trained_model(model_path):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Collect predictions
def evaluate_model(model, dataloader):
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

# Plot confusion matrix
def plot_cm(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Songket Motif Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

# Main execution
if __name__ == "__main__":
    MODEL_FILE = "songket_motif_model.pth"
    
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found. Train the model first!")
    else:
        print(f"🧠 Loading model from {MODEL_FILE}...")
        model = load_trained_model(MODEL_FILE)
        
        print("📊 Evaluating on validation set...")
        all_labels, all_preds = evaluate_model(model, val_loader)
        
        # Calculate accuracy
        accuracy = np.sum(all_labels == all_preds) / len(all_labels)
        print(f"📈 Validation Accuracy: {accuracy*100:.2f}%")
        
        # Plot confusion matrix
        plot_cm(all_labels, all_preds, class_names)
