"""
Fine-tuning script to study the effect of learning rate on ResNet50 training.
Tests 3 different learning rates: 0.0001, 0.001, 0.01
Logs training metrics (loss, accuracy) for each epoch.
Saves results to JSON for analysis and visualization.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from datetime import datetime

# ============================================================================
# 1. SETUP PATHS & DATA
# ============================================================================
data_dir = "dataset/final_split"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Safety check
for d in [train_dir, val_dir]:
    if not os.path.exists(d) or len(os.listdir(d)) == 0:
        print(f"❌ ERROR: {d} is missing or empty.")
        print("👉 Please run split_augmented_dataset.py first!")
        exit()

# Data augmentation
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

# Load dataset
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"✅ Found classes: {class_names}")
print(f"✅ Training samples: {len(image_datasets['train'])}")
print(f"✅ Validation samples: {len(image_datasets['val'])}")

# ============================================================================
# 2. TRAINING FUNCTION
# ============================================================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, lr_value):
    """
    Train model and log metrics for each epoch.
    
    Returns:
        history: dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': list(range(num_epochs))
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Learning Rate: {lr_value}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1:3d}/{len(dataloaders['train']):3d} | "
                      f"Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(image_datasets['train'])
        train_acc = running_corrects.double() / len(image_datasets['train'])
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(image_datasets['val'])
        val_acc = running_corrects.double() / len(image_datasets['val'])
        
        # Store metrics
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1:2d} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Track best validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"   ✨ New best validation accuracy: {best_acc:.4f}")
    
    return history


# ============================================================================
# 3. MAIN EXECUTION: TEST DIFFERENT LEARNING RATES
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {device}")

# Define learning rates to test
learning_rates = [0.0001, 0.001, 0.01]
num_epochs = 30

# Store all results
all_results = {
    'timestamp': datetime.now().isoformat(),
    'device': str(device),
    'num_epochs': num_epochs,
    'batch_size': 16,
    'learning_rates': {},
    'summary_table': []
}

print(f"\n{'='*60}")
print(f"LEARNING RATE FINE-TUNING STUDY - ResNet50")
print(f"{'='*60}")
print(f"Testing {len(learning_rates)} learning rates over {num_epochs} epochs")
print(f"Learning Rates: {learning_rates}")
print(f"{'='*60}\n")

# Train for each learning rate
for lr in learning_rates:
    print(f"\n\n{'█'*60}")
    print(f"█ TRAINING WITH LEARNING RATE = {lr}")
    print(f"{'█'*60}\n")
    
    # Create fresh model for each learning rate
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Unfreeze last layer for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': lr * 0.1},  # Lower rate for backbone
        {'params': model.fc.parameters(), 'lr': lr}              # Higher rate for classifier
    ])
    
    # Train model
    history = train_model(model, dataloaders, criterion, optimizer, 
                         num_epochs, device, lr)
    
    # Save model checkpoint
    model_name = f"songket_resnet50_lr_{lr}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"\n✅ Model saved: {model_name}")
    
    # Store results
    all_results['learning_rates'][str(lr)] = history
    
    # Calculate summary stats
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    
    all_results['summary_table'].append({
        'learning_rate': lr,
        'final_train_acc': float(final_train_acc),
        'final_val_acc': float(final_val_acc),
        'best_val_acc': float(best_val_acc),
        'best_val_epoch': best_val_epoch,
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1])
    })

# ============================================================================
# 4. SAVE RESULTS
# ============================================================================
results_file = "learning_rate_study_results.json"
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\n{'='*60}")
print(f"📊 RESULTS SAVED")
print(f"{'='*60}")
print(f"Results file: {results_file}")

# Print summary table
print(f"\n{'='*60}")
print(f"SUMMARY TABLE")
print(f"{'='*60}")
print(f"{'LR':>10} | {'Final Train Acc':>15} | {'Final Val Acc':>15} | {'Best Val Acc':>12} | {'Best Epoch':>10}")
print(f"{'-'*70}")

for row in all_results['summary_table']:
    print(f"{row['learning_rate']:>10.4f} | {row['final_train_acc']:>15.4f} | "
          f"{row['final_val_acc']:>15.4f} | {row['best_val_acc']:>12.4f} | {row['best_val_epoch']:>10d}")

print(f"{'='*60}")
print(f"\n✅ All training completed! Results saved to: {results_file}")
print(f"📝 Next: Run visualize_lr_study.py to create plots")
