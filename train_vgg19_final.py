"""
Training script for Songket motif classifier using VGG19-BN.
Trains on augmented images from dataset/final_split/train and val.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 1. SETUP PATHS - Use augmented data in the 'final_split' folders
data_dir = "dataset/final_split"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Safety check
for d in [train_dir, val_dir]:
    if not os.path.exists(d) or len(os.listdir(d)) == 0:
        print(f"❌ ERROR: {d} is missing or empty.")
        print("👉 Please run split_augmented_dataset.py first!")
        exit()

# 2. DATA AUGMENTATION
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. LOAD AUGMENTED DATASET
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=(x == 'train'))
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
print(f"✅ Found classes: {class_names}")

# 4. PREPARE THE MODEL (VGG19-BN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

# Freeze all feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final classifier layer for motif classes
num_classes = len(class_names)
# VGG19-BN's classifier has 7 layers; replace the last one (index 6)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)

# 5. DEFINE TRAINING RULES
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
num_epochs = 60
best_val_acc = 0.0
best_state_dict = None

# 6. TRAINING LOOP
print(f"🚀 Training VGG19-BN started on {device}...")
for epoch in range(num_epochs):
    epoch_metrics = {}
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_metrics[phase] = (epoch_loss, epoch_acc.item())
        print(f"Epoch {epoch + 1:02d}/{num_epochs} | {phase.capitalize():5s} Loss: {epoch_loss:.4f} | {phase.capitalize():5s} Acc: {epoch_acc:.4f}")

    scheduler.step()

    val_acc = epoch_metrics['val'][1]
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# 7. SAVE YOUR MODEL
torch.save(model.state_dict(), "songket_motif_vgg19_final.pth")
print(f"✅ Training Complete. Best val accuracy: {best_val_acc * 100:.2f}%")
print("✅ Model saved as 'songket_motif_vgg19_final.pth'")
