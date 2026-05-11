"""
Cross-model learning-rate comparison for four models:
AlexNet, GoogLeNet, VGG19, ResNet50.

Reads existing LR-study JSON files and generates unified plots for thesis writing.
"""

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = os.path.join("results", "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STUDY_FILES = {
    "AlexNet": os.path.join("lr_study_results", "alexnet", "learning_rate_study_results.json"),
    "GoogLeNet": os.path.join("lr_study_results", "googlenet", "learning_rate_study_results.json"),
    "VGG19": os.path.join("lr_study_results", "vgg19", "learning_rate_study_results.json"),
    "ResNet50": os.path.join("lr_study_results", "resnet50", "learning_rate_study_results.json"),
}

MODEL_COLORS = {
    "AlexNet": "#9467bd",
    "GoogLeNet": "#1f77b4",
    "VGG19": "#d62728",
    "ResNet50": "#2ca02c",
}


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing study file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_rows(model_name: str, payload: Dict) -> List[Dict]:
    rows = []
    for lr_key, hist in payload["learning_rates"].items():
        lr = float(lr_key)
        val_acc = hist["val_acc"]
        val_loss = hist["val_loss"]

        best_acc = max(val_acc)
        best_epoch = int(val_acc.index(best_acc) + 1)

        rows.append(
            {
                "model": model_name,
                "lr": lr,
                "epochs": hist.get("epochs", list(range(len(val_acc)))),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "final_val_acc": float(val_acc[-1]),
                "best_val_acc": float(best_acc),
                "best_epoch": best_epoch,
                "final_val_loss": float(val_loss[-1]),
            }
        )

    rows.sort(key=lambda r: r["lr"])
    return rows


def print_summary(rows: List[Dict]) -> None:
    print("\n" + "=" * 108)
    print("{:^108}".format("FOUR-MODEL LEARNING-RATE COMPARISON SUMMARY"))
    print("=" * 108)
    print(
        f"{'Model':<12} {'LR':<9} {'Best Val Acc':<14} {'Best Epoch':<11} "
        f"{'Final Val Acc':<14} {'Final Val Loss':<14}"
    )
    print("-" * 108)

    for r in rows:
        print(
            f"{r['model']:<12} {r['lr']:<9.4f} {r['best_val_acc']:<14.4f} {r['best_epoch']:<11d} "
            f"{r['final_val_acc']:<14.4f} {r['final_val_loss']:<14.4f}"
        )

    print("=" * 108)


def plot_best_acc_vs_lr(rows_by_model: Dict[str, List[Dict]]) -> None:
    plt.figure(figsize=(10, 6))

    for model, rows in rows_by_model.items():
        lrs = [r["lr"] for r in rows]
        vals = [r["best_val_acc"] for r in rows]
        plt.plot(
            lrs,
            vals,
            marker="o",
            linewidth=2.4,
            markersize=7,
            label=model,
            color=MODEL_COLORS[model],
        )

        for x, y in zip(lrs, vals):
            plt.text(x, y + 0.008, f"{y:.3f}", fontsize=8, ha="center", color=MODEL_COLORS[model])

    plt.xscale("log")
    plt.xticks([1e-4, 1e-3, 1e-2], ["0.0001", "0.001", "0.01"])
    plt.ylim(0.75, 1.01)
    plt.grid(alpha=0.25)
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Best Validation Accuracy vs Learning Rate (4 Models)", fontsize=14, fontweight="bold")
    plt.legend(frameon=False)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "lr_compare_4models_best_val_acc.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_final_acc_vs_lr(rows_by_model: Dict[str, List[Dict]]) -> None:
    plt.figure(figsize=(10, 6))

    for model, rows in rows_by_model.items():
        lrs = [r["lr"] for r in rows]
        vals = [r["final_val_acc"] for r in rows]
        plt.plot(
            lrs,
            vals,
            marker="s",
            linewidth=2.4,
            markersize=7,
            label=model,
            color=MODEL_COLORS[model],
        )

    plt.xscale("log")
    plt.xticks([1e-4, 1e-3, 1e-2], ["0.0001", "0.001", "0.01"])
    plt.ylim(0.75, 1.01)
    plt.grid(alpha=0.25)
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Final Validation Accuracy")
    plt.title("Final Validation Accuracy vs Learning Rate (4 Models)", fontsize=14, fontweight="bold")
    plt.legend(frameon=False)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "lr_compare_4models_final_val_acc.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_convergence_per_lr(rows_by_model: Dict[str, List[Dict]], metric_key: str, y_label: str, title: str, out_name: str) -> None:
    lr_values = [1e-4, 1e-3, 1e-2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)

    for i, lr in enumerate(lr_values):
        ax = axes[i]
        for model, rows in rows_by_model.items():
            row = next(r for r in rows if abs(r["lr"] - lr) < 1e-12)
            epochs = [e + 1 for e in row["epochs"]]
            ax.plot(epochs, row[metric_key], linewidth=2.1, label=model, color=MODEL_COLORS[model])

        ax.set_title(f"LR = {lr:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(y_label)
    axes[0].legend(frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_heatmap(rows: List[Dict]) -> None:
    row_labels = [f"{r['model']} | LR={r['lr']:.4f}" for r in rows]
    data = np.array([[r["best_val_acc"], r["final_val_acc"], r["final_val_loss"], r["best_epoch"]] for r in rows], dtype=float)

    norm = data.copy()
    for c in range(norm.shape[1]):
        col = norm[:, c]
        c_min = float(col.min())
        c_max = float(col.max())
        if c_max > c_min:
            norm[:, c] = (col - c_min) / (c_max - c_min)
        else:
            norm[:, c] = 0.5

    plt.figure(figsize=(11.5, 8))
    sns.heatmap(
        norm,
        annot=data,
        fmt=".4f",
        cmap="YlGnBu",
        yticklabels=row_labels,
        xticklabels=["Best Val Acc", "Final Val Acc", "Final Val Loss", "Best Epoch"],
    )
    plt.title("Learning-Rate Study Heatmap Across 4 Models", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "lr_compare_4models_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    rows_by_model: Dict[str, List[Dict]] = {}

    for model, path in STUDY_FILES.items():
        payload = load_json(path)
        rows_by_model[model] = extract_model_rows(model, payload)

    # Validate all models share the same LR set for fair visual comparison
    base_lrs = [r["lr"] for r in rows_by_model["AlexNet"]]
    for model, rows in rows_by_model.items():
        lrs = [r["lr"] for r in rows]
        if lrs != base_lrs:
            raise RuntimeError(f"LR set mismatch for {model}. Expected {base_lrs}, got {lrs}")

    all_rows = []
    for model in ["AlexNet", "GoogLeNet", "VGG19", "ResNet50"]:
        all_rows.extend(rows_by_model[model])

    print_summary(all_rows)
    plot_best_acc_vs_lr(rows_by_model)
    plot_final_acc_vs_lr(rows_by_model)
    plot_convergence_per_lr(
        rows_by_model,
        metric_key="val_acc",
        y_label="Validation Accuracy",
        title="Validation Accuracy Convergence by LR (4 Models)",
        out_name="lr_compare_4models_val_acc_convergence.png",
    )
    plot_convergence_per_lr(
        rows_by_model,
        metric_key="val_loss",
        y_label="Validation Loss",
        title="Validation Loss Convergence by LR (4 Models)",
        out_name="lr_compare_4models_val_loss_convergence.png",
    )
    plot_heatmap(all_rows)

    print("\nDone. Four-model LR comparison plots are in results/comparison/.")


if __name__ == "__main__":
    main()
