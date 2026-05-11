"""
Compare the two modified Songket motif models (GoogLeNet and VGG19-BN)
and generate focused analysis graphs.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchvision import datasets, models, transforms

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["bunga_pecah_lapan", "pucuk_rebung", "tampuk_manggis"]
DATA_DIR = "dataset/final_split"
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join("results", "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TEST_DATASET = datasets.ImageFolder(TEST_DIR, TEST_TRANSFORMS)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATASET, batch_size=16, shuffle=False)


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    labels: np.ndarray
    preds: np.ndarray


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


def load_vgg19_bn(model_path: str) -> torch.nn.Module:
    model = models.vgg19_bn()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_model(model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, labels in TEST_LOADER:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_result(name: str, labels: np.ndarray, preds: np.ndarray) -> ModelResult:
    accuracy = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        labels, preds, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=np.arange(len(CLASS_NAMES)))

    return ModelResult(
        name=name,
        accuracy=accuracy,
        precision_macro=p_macro,
        recall_macro=r_macro,
        f1_macro=f1_macro,
        confusion=cm,
        per_class_precision=p_cls,
        per_class_recall=r_cls,
        per_class_f1=f1_cls,
        labels=labels,
        preds=preds,
    )


def plot_overall_metric_lines(results: List[ModelResult]) -> None:
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metric_labels))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = {"GoogLeNet": "#1f77b4", "VGG19-BN": "#d62728"}

    for result in results:
        values = [result.accuracy, result.precision_macro, result.recall_macro, result.f1_macro]
        color = colors.get(result.name, "#444444")
        ax.plot(x, values, marker="o", linewidth=2.6, markersize=8, label=result.name, color=color)

        for xi, yi in zip(x, values):
            ax.text(xi, yi + 0.015, f"{yi:.3f}", ha="center", va="bottom", fontsize=9, color=color)

    ax.set_title("Modified Model Comparison (Macro Metrics)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "modified_models_macro_metrics_line.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_per_class_grouped_bars(results: List[ModelResult]) -> None:
    # One subplot per metric: precision, recall, F1
    pretty_classes = [c.replace("_", " ").title() for c in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    metric_specs = [
        ("Precision", "per_class_precision", "#4c78a8", "#f58518"),
        ("Recall", "per_class_recall", "#54a24b", "#e45756"),
        ("F1-Score", "per_class_f1", "#72b7b2", "#b279a2"),
    ]

    for ax, (title, attr, c1, c2) in zip(axes, metric_specs):
        r0 = getattr(results[0], attr)
        r1 = getattr(results[1], attr)

        ax.bar(x - width / 2, r0, width, label=results[0].name, color=c1)
        ax.bar(x + width / 2, r1, width, label=results[1].name, color=c2)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_classes, rotation=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="upper center")

    fig.suptitle("Per-Class Comparison: GoogLeNet vs VGG19-BN", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "modified_models_per_class_bars.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_confusion_matrices(results: List[ModelResult]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for ax, result in zip(axes, results):
        sns.heatmap(
            result.confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{result.name} Confusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "modified_models_confusion_matrices.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_error_overlap(googlenet: ModelResult, vgg19: ModelResult) -> None:
    g_correct = googlenet.preds == googlenet.labels
    v_correct = vgg19.preds == vgg19.labels

    both_correct = int(np.sum(g_correct & v_correct))
    g_only_correct = int(np.sum(g_correct & ~v_correct))
    v_only_correct = int(np.sum(~g_correct & v_correct))
    both_wrong = int(np.sum(~g_correct & ~v_correct))

    labels = ["Both Correct", "GoogLeNet Only", "VGG19-BN Only", "Both Wrong"]
    values = [both_correct, g_only_correct, v_only_correct, both_wrong]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Prediction Agreement and Error Overlap", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Test Images")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", va="bottom")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "modified_models_error_overlap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def print_summary(results: List[ModelResult]) -> None:
    print("\n" + "=" * 74)
    print("{:^74}".format("MODIFIED MODEL COMPARISON SUMMARY"))
    print("=" * 74)
    print(f"{'Model':<14} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 74)

    for r in results:
        print(
            f"{r.name:<14} {r.accuracy:<12.4f} {r.precision_macro:<12.4f} "
            f"{r.recall_macro:<12.4f} {r.f1_macro:<12.4f}"
        )

    print("=" * 74)
    delta = results[0].accuracy - results[1].accuracy
    print(f"Accuracy gap ({results[0].name} - {results[1].name}): {delta:+.4f}")


def main() -> None:
    model_files: Dict[str, str] = {
        "GoogLeNet": "songket_motif_googlenet_final.pth",
        "VGG19-BN": "songket_motif_vgg19_final.pth",
    }

    missing = [name for name, path in model_files.items() if not os.path.exists(path)]
    if missing:
        print("❌ Missing model files:")
        for name in missing:
            print(f"   - {name}: {model_files[name]}")
        return

    print(f"🧠 Evaluating modified models on {DEVICE}...")

    g_model = load_googlenet(model_files["GoogLeNet"])
    g_labels, g_preds = evaluate_model(g_model)
    g_result = compute_result("GoogLeNet", g_labels, g_preds)

    v_model = load_vgg19_bn(model_files["VGG19-BN"])
    v_labels, v_preds = evaluate_model(v_model)
    v_result = compute_result("VGG19-BN", v_labels, v_preds)

    results = [g_result, v_result]

    print_summary(results)
    plot_overall_metric_lines(results)
    plot_per_class_grouped_bars(results)
    plot_confusion_matrices(results)
    plot_error_overlap(g_result, v_result)

    print("\n✨ Done. All modified-model comparison plots are in results/comparison/")


if __name__ == "__main__":
    main()
