"""
Training script for Songket motif classifier using GoogLeNet.
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

# 4. MODEL: GoogLeNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layers for motif classes (main + auxiliary heads)
num_classes = len(image_datasets['train'].classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
if getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
    model.aux1.fc = nn.Linear(model.aux1.fc.in_features, num_classes)
if getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
    model.aux2.fc = nn.Linear(model.aux2.fc.in_features, num_classes)
model = model.to(device)

# 5. LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
# Train only the classification heads
head_params = [p for p in model.fc.parameters()]
if getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
    head_params += [p for p in model.aux1.fc.parameters()]
if getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
    head_params += [p for p in model.aux2.fc.parameters()]
optimizer = optim.Adam(head_params, lr=0.001)
best_val_acc = 0.0
best_epoch = 0

# 6. TRAINING LOOP
print(f"🚀 Training GoogLeNet started on {device}...")
for epoch in range(30):  # Adjust as needed
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                # In training with aux_logits=True, outputs is InceptionOutputs(logits, aux_logits)
                if phase == 'train' and hasattr(outputs, 'logits'):
                    main_logits = outputs.logits
                    aux_logits = outputs.aux_logits
                    _, preds = torch.max(main_logits, 1)
                    loss = criterion(main_logits, labels) + 0.3 * criterion(aux_logits, labels)
                else:
                    # Evaluation returns a Tensor
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        epoch_loss = running_loss / len(image_datasets[phase])
        print(f"Epoch {epoch + 1:02d}/30 | {phase.capitalize():5s} Loss: {epoch_loss:.4f} | {phase.capitalize():5s} Acc: {epoch_acc:.4f}")

        if phase == 'val' and epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "songket_motif_googlenet_final.pth")

# 7. SAVE MODEL
print(f"✅ Training Complete. Best val accuracy: {best_val_acc * 100:.2f}% at epoch {best_epoch}")
print("✅ Model saved as 'songket_motif_googlenet_final.pth'")
