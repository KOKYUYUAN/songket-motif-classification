import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['bunga_pecah_lapan', 'pucuk_rebung', 'tampuk_manggis']
data_dir = "dataset/final_split"
test_dir = os.path.join(data_dir, "test")

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

def load_model(model_path):
    state = torch.load(model_path, map_location=device)
    has_aux = any((k.startswith("aux1") or k.startswith("aux2")) for k in state.keys())
    model = models.googlenet(aux_logits=has_aux, init_weights=True)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    if has_aux and getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
        model.aux1.fc = nn.Linear(model.aux1.fc.in_features, len(class_names))
    if has_aux and getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
        model.aux2.fc = nn.Linear(model.aux2.fc.in_features, len(class_names))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

print("Testing songket_motif_googlenet.pth (Dec 25 - original)...")
model_old = load_model("songket_motif_googlenet.pth")

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model_old(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels.to(device)).sum().item()
        total += labels.size(0)

acc_old = 100 * correct / total
print(f"Old model accuracy: {acc_old:.2f}%\n")

print("Testing songket_motif_googlenet_final.pth (Apr 18 - retrained)...")
model_new = load_model("songket_motif_googlenet_final.pth")

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model_new(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels.to(device)).sum().item()
        total += labels.size(0)

acc_new = 100 * correct / total
print(f"New model accuracy: {acc_new:.2f}%")
