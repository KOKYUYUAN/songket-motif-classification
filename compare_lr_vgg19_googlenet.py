"""
Cross-model learning-rate comparison for VGG19 and GoogLeNet.

Reads existing LR-study JSON files and generates side-by-side analysis plots:
1) Best validation accuracy vs learning rate
2) Final validation accuracy vs learning rate
3) Validation-loss convergence at each learning rate
4) Validation-accuracy convergence at each learning rate
5) Summary heatmap of key metrics
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BASE_DIR = "lr_study_results"
VGG_JSON = os.path.join(BASE_DIR, "vgg19", "learning_rate_study_results.json")
GOOGLENET_JSON = os.path.join(BASE_DIR, "googlenet", "learning_rate_study_results.json")
OUTPUT_DIR = os.path.join("results", "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_lr_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing LR study file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_lr_rows(study: Dict, model_name: str) -> List[Dict]:
    rows: List[Dict] = []

    for lr_key, hist in study["learning_rates"].items():
        lr = float(lr_key)
        val_acc = hist["val_acc"]
        val_loss = hist["val_loss"]

        best_val_acc = max(val_acc)
        best_epoch = val_acc.index(best_val_acc) + 1

        rows.append(
            {
                "model": model_name,
                "lr": lr,
                "epochs": hist["epochs"],
                "val_acc": val_acc,
                "val_loss": val_loss,
                "final_val_acc": val_acc[-1],
                "final_val_loss": val_loss[-1],
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }
        )

    rows.sort(key=lambda r: r["lr"])
    return rows


def print_summary_table(rows: List[Dict]) -> None:
    print("\n" + "=" * 96)
    print("{:^96}".format("VGG19 vs GOOGLENET LEARNING-RATE SUMMARY"))
    print("=" * 96)
    print(
        f"{'Model':<12} {'LR':<10} {'Final Val Acc':<15} "
        f"{'Best Val Acc':<14} {'Best Epoch':<11} {'Final Val Loss':<15}"
    )
    print("-" * 96)

    for r in rows:
        print(
            f"{r['model']:<12} {r['lr']:<10.4f} {r['final_val_acc']:<15.4f} "
            f"{r['best_val_acc']:<14.4f} {r['best_epoch']:<11d} {r['final_val_loss']:<15.4f}"
        )

    print("=" * 96)


def plot_best_and_final_acc(vgg_rows: List[Dict], g_rows: List[Dict]) -> None:
    lrs = [r["lr"] for r in vgg_rows]
    x = np.arange(len(lrs))
    width = 0.36

    v_best = [r["best_val_acc"] for r in vgg_rows]
    g_best = [r["best_val_acc"] for r in g_rows]
    v_final = [r["final_val_acc"] for r in vgg_rows]
    g_final = [r["final_val_acc"] for r in g_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)

    ax = axes[0]
    ax.bar(x - width / 2, v_best, width, color="#d62728", label="VGG19")
    ax.bar(x + width / 2, g_best, width, color="#1f77b4", label="GoogLeNet")
    ax.set_title("Best Validation Accuracy by LR", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.4f}" for lr in lrs])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x - width / 2, v_final, width, color="#ff9896", label="VGG19")
    ax.bar(x + width / 2, g_final, width, color="#9ecae1", label="GoogLeNet")
    ax.set_title("Final Validation Accuracy by LR", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.4f}" for lr in lrs])
    ax.set_xlabel("Learning Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "lr_compare_vgg19_googlenet_accuracy.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_convergence(vgg_rows: List[Dict], g_rows: List[Dict], metric_key: str, title: str, y_label: str, out_name: str) -> None:
    lrs = [r["lr"] for r in vgg_rows]
    n = len(lrs)

    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]

    for i, lr in enumerate(lrs):
        vr = vgg_rows[i]
        gr = g_rows[i]
        epochs = [e + 1 for e in vr["epochs"]]

        axes[i].plot(epochs, vr[metric_key], color="#d62728", linewidth=2.2, label="VGG19")
        axes[i].plot(epochs, gr[metric_key], color="#1f77b4", linewidth=2.2, label="GoogLeNet")
        axes[i].set_title(f"LR={lr:.4f}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].grid(alpha=0.25)

    axes[0].set_ylabel(y_label)
    axes[0].legend(frameon=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def build_heatmap_data(vgg_rows: List[Dict], g_rows: List[Dict]) -> Tuple[np.ndarray, List[str], List[str]]:
    # rows: model@lr, cols: [best_val_acc, final_val_acc, final_val_loss]
    data_rows = []
    row_labels = []

    for rows, model in [(vgg_rows, "VGG19"), (g_rows, "GoogLeNet")]:
        for r in rows:
            data_rows.append([r["best_val_acc"], r["final_val_acc"], r["final_val_loss"]])
            row_labels.append(f"{model} | LR={r['lr']:.4f}")

    col_labels = ["Best Val Acc", "Final Val Acc", "Final Val Loss"]
    return np.array(data_rows), row_labels, col_labels


def plot_summary_heatmap(vgg_rows: List[Dict], g_rows: List[Dict]) -> None:
    data, row_labels, col_labels = build_heatmap_data(vgg_rows, g_rows)

    # Normalize each column for visual comparability
    norm = data.copy()
    for c in range(norm.shape[1]):
        col = norm[:, c]
        c_min = float(col.min())
        c_max = float(col.max())
        if c_max > c_min:
            norm[:, c] = (col - c_min) / (c_max - c_min)
        else:
            norm[:, c] = 0.5

    plt.figure(figsize=(10.5, 6.2))
    sns.heatmap(norm, annot=data, fmt=".4f", cmap="YlGnBu", xticklabels=col_labels, yticklabels=row_labels)
    plt.title("LR Study Summary Heatmap: VGG19 vs GoogLeNet", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "lr_compare_vgg19_googlenet_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    print("Loading LR-study results...")
    vgg_json = load_lr_json(VGG_JSON)
    googlenet_json = load_lr_json(GOOGLENET_JSON)

    vgg_rows = extract_lr_rows(vgg_json, "VGG19")
    g_rows = extract_lr_rows(googlenet_json, "GoogLeNet")

    v_lrs = [r["lr"] for r in vgg_rows]
    g_lrs = [r["lr"] for r in g_rows]
    if v_lrs != g_lrs:
        raise RuntimeError("VGG19 and GoogLeNet LR sets differ. Use matching learning rates for fair comparison.")

    all_rows = vgg_rows + g_rows
    print_summary_table(all_rows)

    plot_best_and_final_acc(vgg_rows, g_rows)
    plot_convergence(
        vgg_rows,
        g_rows,
        metric_key="val_loss",
        title="Validation Loss Convergence by Learning Rate",
        y_label="Validation Loss",
        out_name="lr_compare_vgg19_googlenet_val_loss_convergence.png",
    )
    plot_convergence(
        vgg_rows,
        g_rows,
        metric_key="val_acc",
        title="Validation Accuracy Convergence by Learning Rate",
        y_label="Validation Accuracy",
        out_name="lr_compare_vgg19_googlenet_val_acc_convergence.png",
    )
    plot_summary_heatmap(vgg_rows, g_rows)

    print("\nDone. LR comparison plots are saved under results/comparison/.")


if __name__ == "__main__":
    main()
