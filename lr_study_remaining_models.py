"""
Learning-rate study for remaining models: AlexNet, GoogLeNet, VGG19.

This script:
1. Runs LR experiments for each model (default LRs: 0.0001, 0.001, 0.01)
2. Trains for N epochs per LR (default: 30)
3. Saves per-model JSON metrics and checkpoints
4. Generates 5 analysis plots per model

Output structure:
  lr_study_results/
    alexnet/
      learning_rate_study_results.json
      checkpoints/
      plots/
    googlenet/
      learning_rate_study_results.json
      checkpoints/
      plots/
    vgg19/
      learning_rate_study_results.json
      checkpoints/
      plots/
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="LR study for AlexNet/GoogLeNet/VGG19")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["alexnet", "googlenet", "vgg19"],
        choices=["alexnet", "googlenet", "vgg19"],
        help="Models to run",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per LR")
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[0.0001, 0.001, 0.01],
        help="Learning rates to test",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output-root",
        type=str,
        default="lr_study_results",
        help="Root folder for all LR-study outputs",
    )
    return parser.parse_args()


def build_dataloaders(batch_size):
    data_dir = "dataset/final_split"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    for d in [train_dir, val_dir]:
        if not os.path.exists(d) or len(os.listdir(d)) == 0:
            raise RuntimeError(f"{d} is missing or empty. Run split_augmented_dataset.py first.")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, data_transforms["val"]),
    }
    dataloaders = {
        split: torch.utils.data.DataLoader(image_datasets[split], batch_size=batch_size, shuffle=True)
        for split in ["train", "val"]
    }
    return image_datasets, dataloaders


def init_model_and_optimizer(model_name, num_classes, lr, device):
    criterion = nn.CrossEntropyLoss()

    if model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        for p in model.features.parameters():
            p.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=lr)

    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        for p in model.features.parameters():
            p.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        optimizer = optim.Adam(model.classifier[6].parameters(), lr=lr)

    elif model_name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
            model.aux1.fc = nn.Linear(model.aux1.fc.in_features, num_classes)
        if getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
            model.aux2.fc = nn.Linear(model.aux2.fc.in_features, num_classes)

        head_params = list(model.fc.parameters())
        if getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
            head_params += list(model.aux1.fc.parameters())
        if getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
            head_params += list(model.aux2.fc.parameters())

        optimizer = optim.Adam(head_params, lr=lr)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(device)
    return model, criterion, optimizer


def run_epoch(model, dataloaders, image_datasets, criterion, optimizer, device, model_name, phase):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)

            if model_name == "googlenet" and is_train and hasattr(outputs, "logits"):
                main_logits = outputs.logits
                aux_logits = outputs.aux_logits
                _, preds = torch.max(main_logits, 1)
                loss = criterion(main_logits, labels) + 0.3 * criterion(aux_logits, labels)
            else:
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_datasets[phase])
    epoch_acc = running_corrects.double() / len(image_datasets[phase])
    return float(epoch_loss), float(epoch_acc)


def train_for_lr(model_name, lr, epochs, dataloaders, image_datasets, num_classes, device):
    model, criterion, optimizer = init_model_and_optimizer(model_name, num_classes, lr, device)

    history = {
        "epochs": list(range(epochs)),
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model, dataloaders, image_datasets, criterion, optimizer, device, model_name, "train"
        )
        val_loss, val_acc = run_epoch(
            model, dataloaders, image_datasets, criterion, optimizer, device, model_name, "val"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{model_name}] LR={lr} | Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

    return model, history


def save_plots(summary_rows, data, plots_dir, model_name):
    os.makedirs(plots_dir, exist_ok=True)

    sorted_lrs = sorted(data.keys())
    colors = plt.cm.tab10(range(len(sorted_lrs)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name.upper()} - Effect of Learning Rate on Loss", fontsize=16, fontweight="bold")

    ax = axes[0]
    for idx, lr in enumerate(sorted_lrs):
        h = data[lr]
        ax.plot(h["epochs"], h["train_loss"], marker="o", linewidth=2, markersize=4, color=colors[idx], label=f"LR={lr}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss vs Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for idx, lr in enumerate(sorted_lrs):
        h = data[lr]
        ax.plot(h["epochs"], h["val_loss"], marker="s", linewidth=2, markersize=4, color=colors[idx], label=f"LR={lr}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss vs Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_loss_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name.upper()} - Effect of Learning Rate on Accuracy", fontsize=16, fontweight="bold")

    ax = axes[0]
    for idx, lr in enumerate(sorted_lrs):
        h = data[lr]
        ax.plot(h["epochs"], h["train_acc"], marker="o", linewidth=2, markersize=4, color=colors[idx], label=f"LR={lr}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")
    ax.set_title("Training Accuracy vs Epoch")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for idx, lr in enumerate(sorted_lrs):
        h = data[lr]
        ax.plot(h["epochs"], h["val_acc"], marker="s", linewidth=2, markersize=4, color=colors[idx], label=f"LR={lr}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy vs Epoch")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_accuracy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    lr_values = [r["learning_rate"] for r in summary_rows]
    final_train = [r["final_train_acc"] for r in summary_rows]
    final_val = [r["final_val_acc"] for r in summary_rows]
    best_val = [r["best_val_acc"] for r in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name.upper()} - Final Performance Comparison", fontsize=16, fontweight="bold")

    ax = axes[0]
    x = list(range(len(lr_values)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], final_train, width, label="Final Train Acc", alpha=0.85, color="steelblue")
    ax.bar([i + width / 2 for i in x], final_val, width, label="Final Val Acc", alpha=0.85, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.4f}" for lr in lr_values])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Accuracy by LR")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    ax = axes[1]
    bars = ax.bar(range(len(best_val)), best_val, color=colors[: len(best_val)], alpha=0.85)
    ax.set_xticks(range(len(best_val)))
    ax.set_xticklabels([f"{lr:.4f}" for lr in lr_values])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Peak Validation Accuracy")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, best_val):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_final_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"{model_name.upper()} - Convergence Speed", fontsize=16, fontweight="bold")
    for idx, lr in enumerate(sorted_lrs):
        h = data[lr]
        ax.plot(h["epochs"], h["val_acc"], marker="o", linewidth=2.5, markersize=5, color=colors[idx], label=f"LR={lr}")
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="80% Threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy Progression")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_convergence_speed.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"{model_name.upper()} - Overfitting Analysis", fontsize=16, fontweight="bold")
    for idx, lr in enumerate(sorted_lrs):
        h = data[lr]
        gap = [t - v for t, v in zip(h["train_acc"], h["val_acc"])]
        ax.plot(h["epochs"], gap, marker="o", linewidth=2.5, markersize=5, color=colors[idx], label=f"LR={lr}")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Acc - Val Acc")
    ax.set_title("Generalization Gap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "05_overfitting_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_model_study(model_name, args, image_datasets, dataloaders, device):
    num_classes = len(image_datasets["train"].classes)

    model_root = os.path.join(args.output_root, model_name)
    checkpoints_dir = os.path.join(model_root, "checkpoints")
    plots_dir = os.path.join(model_root, "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "device": str(device),
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rates": {},
        "summary_table": [],
    }

    print("\n" + "=" * 70)
    print(f"Starting LR study for {model_name.upper()}")
    print("=" * 70)

    for lr in args.learning_rates:
        print("\n" + "#" * 70)
        print(f"Model={model_name.upper()} | LR={lr}")
        print("#" * 70)

        model, history = train_for_lr(
            model_name=model_name,
            lr=lr,
            epochs=args.epochs,
            dataloaders=dataloaders,
            image_datasets=image_datasets,
            num_classes=num_classes,
            device=device,
        )

        ckpt_name = f"songket_{model_name}_lr_{lr}.pth"
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        results["learning_rates"][str(lr)] = history

        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1

        results["summary_table"].append(
            {
                "learning_rate": lr,
                "final_train_acc": float(history["train_acc"][-1]),
                "final_val_acc": float(history["val_acc"][-1]),
                "best_val_acc": float(best_val_acc),
                "best_val_epoch": int(best_epoch),
                "final_train_loss": float(history["train_loss"][-1]),
                "final_val_loss": float(history["val_loss"][-1]),
            }
        )

    json_path = os.path.join(model_root, "learning_rate_study_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON: {json_path}")

    data = {float(k): v for k, v in results["learning_rates"].items()}
    save_plots(results["summary_table"], data, plots_dir, model_name)
    print(f"Saved 5 plots to: {plots_dir}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("LR Study for Remaining Models")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Epochs per LR: {args.epochs}")
    print(f"Learning rates: {args.learning_rates}")
    print(f"Output root: {args.output_root}")

    image_datasets, dataloaders = build_dataloaders(args.batch_size)
    print(f"Classes: {image_datasets['train'].classes}")
    print(f"Train samples: {len(image_datasets['train'])}")
    print(f"Val samples: {len(image_datasets['val'])}")

    for model_name in args.models:
        run_model_study(model_name, args, image_datasets, dataloaders, device)

    print("\n" + "=" * 70)
    print("All requested model studies completed.")
    print("You can use each model's 5 plots directly in thesis section 4.3.")
    print("=" * 70)


if __name__ == "__main__":
    main()
