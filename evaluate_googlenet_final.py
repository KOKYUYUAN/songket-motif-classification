"""
Model performance evaluation script for GoogLeNet.
Evaluates on test set and generates confusion matrix, classification report, and performance graphs.
"""

import os
import time

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from torchvision import datasets, transforms, models
import torch.nn as nn

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['bunga_pecah_lapan', 'pucuk_rebung', 'tampuk_manggis']
data_dir = "dataset/final_split"
test_dir = os.path.join(data_dir, "test")
output_dir = os.path.join("results", "googlenet")
os.makedirs(output_dir, exist_ok=True)

# Data transforms (same as validation in training)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = datasets.ImageFolder(test_dir, test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load trained model
def load_trained_model(model_path):
    """Load GoogLeNet weights, auto-detecting whether aux heads were saved."""
    state = torch.load(model_path, map_location=device)
    # Detect presence of aux heads in checkpoint keys
    has_aux = any((k.startswith("aux1") or k.startswith("aux2")) for k in state.keys())

    model = models.googlenet(aux_logits=has_aux, init_weights=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    if has_aux and getattr(model, "aux1", None) is not None and hasattr(model.aux1, "fc"):
        model.aux1.fc = nn.Linear(model.aux1.fc.in_features, len(class_names))
    if has_aux and getattr(model, "aux2", None) is not None and hasattr(model.aux2, "fc"):
        model.aux2.fc = nn.Linear(model.aux2.fc.in_features, len(class_names))

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

# Collect predictions
def evaluate_model(model, dataloader):
    """
    Enhanced evaluation: tracks labels, predictions,
    probabilities, and latency per image.
    """
    all_labels = []
    all_preds = []
    all_probs = []
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    dataset_size = len(dataloader.dataset)
    avg_latency = total_time_ms / dataset_size if dataset_size > 0 else 0.0
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), avg_latency


def plot_confidence_distribution(all_probs, all_labels):
    """
    Visualize model confidence distribution.
    High confidence on errors can indicate overfitting or data quality issues.
    """
    _ = all_labels
    confidences = np.max(all_probs, axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(
        np.mean(confidences),
        color='red',
        linestyle='dashed',
        linewidth=2,
        label=f'Mean: {np.mean(confidences):.2f}'
    )

    plt.title('Distribution of Prediction Confidence Scores', fontsize=14)
    plt.xlabel('Confidence (Probability)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confidence_distribution_googlenet_final.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confidence distribution saved as '{output_path}'")
    plt.close()

# Plot confusion matrix
def plot_cm(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('GoogLeNet - Songket Motif Confusion Matrix')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix_googlenet_final.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved as '{output_path}'")
    plt.close()

# Plot per-class metrics
def plot_per_class_metrics(all_labels, all_preds, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, zero_division=0)

    pretty_labels = [name.replace('_', ' ').title() for name in class_names]

    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)

        metric_specs = [
            ('Precision', precision, '#C5A059', '#8c6a2f'),
            ('Recall', recall, '#1A237E', '#121858'),
            ('F1-Score', f1, '#546E7A', '#3b4d55'),
        ]

        for ax, (title, values, fill_color, edge_color) in zip(axes, metric_specs):
            bars = ax.bar(pretty_labels, values, color=fill_color, edgecolor=edge_color, linewidth=1.2, width=0.62)
            ax.set_title(title, fontsize=13, fontweight='bold', color='#3f2f25')
            ax.set_ylim(0, 1.02)
            ax.set_ylabel('Score', fontsize=11, fontweight='bold', color='#3f2f25')
            ax.grid(axis='y', alpha=0.28, color='#c9c1b7')
            ax.grid(axis='x', visible=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.tick_params(axis='x', labelrotation=12)
            for tick in ax.get_xticklabels():
                tick.set_fontweight('bold')
                tick.set_color('#2b211a')
                tick.set_fontsize(10)

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.02,
                    f'{value:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    color='#3f2f25'
                )

        fig.suptitle('Per-Class Metric Profile', fontsize=16, fontweight='bold', color='#3f2f25', y=1.04)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_class_metrics_googlenet_final.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Per-class metrics saved as '{output_path}'")
    plt.close()

# Plot accuracy and per-class distribution
def plot_accuracy_distribution(all_labels, all_preds, class_names):
    overall_acc = accuracy_score(all_labels, all_preds)

    per_class_acc = []
    for i in range(len(class_names)):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = np.sum(all_preds[mask] == i) / mask.sum()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0)

    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, axes = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={'width_ratios': [1.0, 1.45]})

        ring_color = '#b08d57' if overall_acc >= 0.8 else '#c27b5f' if overall_acc >= 0.6 else '#7a1f2b'
        axes[0].pie(
            [overall_acc, 1 - overall_acc],
            colors=[ring_color, '#efe9df'],
            startangle=90,
            counterclock=False,
            radius=0.9,
            wedgeprops={'width': 0.30, 'edgecolor': 'white'}
        )
        axes[0].text(
            0,
            0,
            f"{overall_acc*100:.1f}%",
            ha='center',
            va='center',
            fontsize=26,
            fontweight='bold',
            color='#3f2f25',
            fontfamily='DejaVu Serif'
        )
        axes[0].set_title('Overall Accuracy', fontsize=15, pad=18, fontweight='bold', color='#3f2f25')

        y_positions = np.arange(1, len(class_names) + 1)
        class_labels = [name.replace('_', ' ').title() for name in class_names]
        palette = ['#7a1f2b', '#b08d57', '#d9b382']

        axes[1].hlines(y=y_positions, xmin=0, xmax=per_class_acc, color='#cdb79e', alpha=0.9, linewidth=5)
        axes[1].scatter(per_class_acc, y_positions, s=190, color=palette[:len(class_names)], edgecolor='white', linewidth=1.8, zorder=3)

        for y, acc in zip(y_positions, per_class_acc):
            axes[1].text(acc + 0.02, y, f'{acc*100:.1f}%', va='center', fontsize=10, color='#3f2f25', fontweight='bold')

        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(class_labels)
        for label in axes[1].get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('#2b211a')
            label.set_fontsize(11)
        axes[1].set_xlim(0, 1.05)
        axes[1].set_xlabel('Accuracy Score', fontsize=12)
        axes[1].set_title('Accuracy per Motif Category', fontsize=15, pad=18, fontweight='bold', color='#3f2f25')
        axes[1].grid(axis='y', visible=False)
        axes[1].grid(axis='x', alpha=0.25, color='#c9c1b7')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['left'].set_visible(False)

        fig.suptitle('GoogLeNet Performance Dashboard', fontsize=17, fontweight='bold', color='#3f2f25', y=1.02)
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'accuracy_dashboard_googlenet_final.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Accuracy dashboard saved as '{output_path}'")
        plt.close()

# Generate comprehensive performance summary
def generate_performance_summary(all_labels, all_preds, class_names):
    overall_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
    
    print("\n" + "="*70)
    print(" "*15 + "PERFORMANCE SUMMARY - GoogLeNet")
    print("="*70)
    print(f"\n🎯 OVERALL TEST ACCURACY: {overall_acc*100:.2f}%\n")
    print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-"*70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision[i]:<15.4f} {recall[i]:<15.4f} {f1[i]:<15.4f} {support[i]:<10}")
    print("-"*70)
    print(f"{'Macro Average':<20} {precision.mean():<15.4f} {recall.mean():<15.4f} {f1.mean():<15.4f} {support.sum():<10}")
    print("="*70 + "\n")

# Main execution
if __name__ == "__main__":
    MODEL_FILE = "songket_motif_googlenet_final.pth"
    
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: {MODEL_FILE} not found. Train the model first!")
    else:
        print(f"🧠 Loading model from {MODEL_FILE}...")
        model = load_trained_model(MODEL_FILE)
        
        print("📊 Running advanced evaluation on test set...")
        all_labels, all_preds, all_probs, latency = evaluate_model(model, test_loader)
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n📈 Test Accuracy: {accuracy*100:.2f}%")
        print(f"⚡ Inference Latency: {latency:.2f} ms per image")
        
        print("\n📜 CLASSIFICATION REPORT")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
        
        generate_performance_summary(all_labels, all_preds, class_names)
        
        print("📊 Generating performance visualizations...\n")
        plot_confidence_distribution(all_probs, all_labels)
        plot_cm(all_labels, all_preds, class_names)
        plot_per_class_metrics(all_labels, all_preds, class_names)
        plot_accuracy_distribution(all_labels, all_preds, class_names)
        
        print("\n✨ All performance evaluations complete!")
