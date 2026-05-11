"""
Compare four Songket motif classifiers and generate a gradient comparison graph.
Models included:
- GoogLeNet
- ResNet50
- VGG19
- AlexNet

The script evaluates each model on dataset/final_split/test and saves a visual
comparison chart as model_comparison_gradient.png.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import datasets, models, transforms

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['bunga_pecah_lapan', 'pucuk_rebung', 'tampuk_manggis']
DATA_DIR = "dataset/final_split"
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join("results", "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TEST_DATASET = datasets.ImageFolder(TEST_DIR, TEST_TRANSFORMS)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATASET, batch_size=16, shuffle=False)


@dataclass
class ModelResult:
    name: str
    model_path: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate_model(model: torch.nn.Module, dataloader) -> Tuple[np.ndarray, np.ndarray]:
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def load_googlenet(model_path: str) -> torch.nn.Module:
    state = torch.load(model_path, map_location=DEVICE)
    has_aux = any(k.startswith("aux1") or k.startswith("aux2") for k in state.keys())

    model = models.googlenet(aux_logits=has_aux, init_weights=True)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    if has_aux and getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
        model.aux1.fc = nn.Linear(model.aux1.fc.in_features, len(CLASS_NAMES))
    if has_aux and getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
        model.aux2.fc = nn.Linear(model.aux2.fc.in_features, len(CLASS_NAMES))

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def load_resnet50(model_path: str) -> torch.nn.Module:
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_vgg19(model_path: str) -> torch.nn.Module:
    model = models.vgg19_bn()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_alexnet(model_path: str) -> torch.nn.Module:
    model = models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


MODEL_LOADERS: Dict[str, Callable[[str], torch.nn.Module]] = {
    "GoogLeNet": load_googlenet,
    "ResNet50": load_resnet50,
    "VGG19": load_vgg19,
    "AlexNet": load_alexnet,
}

MODEL_FILES: Dict[str, str] = {
    "GoogLeNet": "songket_motif_googlenet_final.pth",
    "ResNet50": "songket_motif_resnet50_final.pth",
    "VGG19": "songket_motif_vgg19_final.pth",
    "AlexNet": "songket_motif_alexnet_final.pth",
}


def compute_metrics(all_labels: np.ndarray, all_preds: np.ndarray) -> Tuple[float, float, float, float]:
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    return accuracy, precision, recall, f1


def plot_gradient_comparison(results: List[ModelResult]) -> None:
    sns.set_style("whitegrid")

    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metric_labels))

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, result in enumerate(results):
        values = [result.accuracy, result.precision, result.recall, result.f1]
        ax.plot(
            x,
            values,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=palette[idx],
            label=result.name,
        )
        for xi, yi in zip(x, values):
            ax.text(
                xi,
                yi + 0.015,
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=palette[idx],
                fontweight="bold",
            )

    ax.set_title("Songket Motif Model Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "model_comparison_line.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Comparison line chart saved as '{output_path}'")


def print_summary_table(results: List[ModelResult]) -> None:
    print("\n" + "=" * 78)
    print("{:^78}".format("MODEL COMPARISON SUMMARY"))
    print("=" * 78)
    print(f"{'Model':<14} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 78)
    for result in results:
        print(
            f"{result.name:<14} {result.accuracy:<12.4f} {result.precision:<12.4f} "
            f"{result.recall:<12.4f} {result.f1:<12.4f}"
        )
    print("=" * 78 + "\n")


def main() -> None:
    missing = [name for name, path in MODEL_FILES.items() if not os.path.exists(path)]
    if missing:
        print("❌ Missing model files:")
        for name in missing:
            print(f"   - {name}: {MODEL_FILES[name]}")
        return

    results: List[ModelResult] = []

    print(f"🧠 Evaluating on {DEVICE}...")
    for model_name, model_path in MODEL_FILES.items():
        print(f"📊 Loading {model_name} from {model_path}...")
        model = MODEL_LOADERS[model_name](model_path)
        all_labels, all_preds = evaluate_model(model, TEST_LOADER)
        accuracy, precision, recall, f1 = compute_metrics(all_labels, all_preds)

        results.append(
            ModelResult(
                name=model_name,
                model_path=model_path,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

        print(f"   -> Accuracy: {accuracy*100:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    print_summary_table(results)
    plot_gradient_comparison(results)
    print("✨ Model comparison complete!")


if __name__ == "__main__":
    main()
