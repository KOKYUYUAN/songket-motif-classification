"""
Visualization script for learning rate study results.
Reads learning_rate_study_results.json and creates comparison plots.
Generates:
  - Training loss comparison
  - Validation loss comparison
  - Training accuracy comparison
  - Validation accuracy comparison
  - Convergence speed comparison
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================================
# 1. LOAD RESULTS
# ============================================================================
results_file = "learning_rate_study_results.json"

if not Path(results_file).exists():
    print(f"❌ ERROR: {results_file} not found!")
    print("👉 Please run finetune_resnet50_lr_study.py first")
    exit()

with open(results_file, 'r') as f:
    results = json.load(f)

learning_rates = list(results['learning_rates'].keys())
print(f"✅ Loaded results for learning rates: {learning_rates}")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
data = {}
for lr_str in learning_rates:
    lr_float = float(lr_str)
    history = results['learning_rates'][lr_str]
    data[lr_float] = history

# Sort by learning rate value
sorted_lrs = sorted(data.keys())

# ============================================================================
# 3. CREATE VISUALIZATIONS
# ============================================================================
colors = plt.cm.tab10(range(len(sorted_lrs)))
line_styles = ['-', '--', '-.', ':']

# Figure 1: Loss Comparison (Training + Validation)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Effect of Learning Rate on Loss', fontsize=16, fontweight='bold')

# Training Loss
ax = axes[0]
for idx, lr in enumerate(sorted_lrs):
    history = data[lr]
    ax.plot(history['epochs'], history['train_loss'], 
            marker='o', label=f'LR={lr}', linewidth=2, 
            color=colors[idx], markersize=4)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Training Loss vs Epoch', fontsize=13)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# Validation Loss
ax = axes[1]
for idx, lr in enumerate(sorted_lrs):
    history = data[lr]
    ax.plot(history['epochs'], history['val_loss'], 
            marker='s', label=f'LR={lr}', linewidth=2,
            color=colors[idx], markersize=4)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss vs Epoch', fontsize=13)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_loss_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 01_loss_comparison.png")
plt.close()

# Figure 2: Accuracy Comparison (Training + Validation)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Effect of Learning Rate on Accuracy', fontsize=16, fontweight='bold')

# Training Accuracy
ax = axes[0]
for idx, lr in enumerate(sorted_lrs):
    history = data[lr]
    ax.plot(history['epochs'], history['train_acc'], 
            marker='o', label=f'LR={lr}', linewidth=2,
            color=colors[idx], markersize=4)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Accuracy', fontsize=12)
ax.set_title('Training Accuracy vs Epoch', fontsize=13)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Validation Accuracy
ax = axes[1]
for idx, lr in enumerate(sorted_lrs):
    history = data[lr]
    ax.plot(history['epochs'], history['val_acc'], 
            marker='s', label=f'LR={lr}', linewidth=2,
            color=colors[idx], markersize=4)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('Validation Accuracy vs Epoch', fontsize=13)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('02_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 02_accuracy_comparison.png")
plt.close()

# Figure 3: Final Performance (Bar Chart)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Final Performance Comparison (After 30 Epochs)', fontsize=16, fontweight='bold')

summary = results['summary_table']
lr_values = [s['learning_rate'] for s in summary]
final_train_accs = [s['final_train_acc'] for s in summary]
final_val_accs = [s['final_val_acc'] for s in summary]
best_val_accs = [s['best_val_acc'] for s in summary]

# Training vs Validation Accuracy
ax = axes[0]
x_pos = range(len(lr_values))
width = 0.35
ax.bar([p - width/2 for p in x_pos], final_train_accs, width, 
       label='Final Train Acc', alpha=0.8, color='steelblue')
ax.bar([p + width/2 for p in x_pos], final_val_accs, width, 
       label='Final Val Acc', alpha=0.8, color='coral')

ax.set_xlabel('Learning Rate', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Final Accuracy by Learning Rate', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{lr:.4f}' for lr in lr_values])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.05])

# Best Validation Accuracy
ax = axes[1]
bars = ax.bar(range(len(summary)), best_val_accs, color=colors[:len(summary)], alpha=0.8)
ax.set_xlabel('Learning Rate', fontsize=12)
ax.set_ylabel('Best Validation Accuracy', fontsize=12)
ax.set_title('Peak Validation Accuracy Achieved', fontsize=13)
ax.set_xticks(range(len(summary)))
ax.set_xticklabels([f'{s["learning_rate"]:.4f}' for s in summary])
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.05])

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, best_val_accs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('03_final_performance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 03_final_performance.png")
plt.close()

# Figure 4: Convergence Speed Analysis
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Convergence Speed: Validation Accuracy Progression', fontsize=16, fontweight='bold')

for idx, lr in enumerate(sorted_lrs):
    history = data[lr]
    ax.plot(history['epochs'], history['val_acc'], 
            marker='o', label=f'LR={lr}', linewidth=2.5,
            color=colors[idx], markersize=6)

# Add reference line for 80% accuracy
ax.axhline(y=0.80, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='80% Threshold')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('How quickly does each learning rate reach target accuracy?', fontsize=13)
ax.legend(loc='best', fontsize=11, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('04_convergence_speed.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 04_convergence_speed.png")
plt.close()

# Figure 5: Overfitting Analysis (Train-Val Gap)
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Overfitting Analysis: Train-Validation Accuracy Gap', fontsize=16, fontweight='bold')

for idx, lr in enumerate(sorted_lrs):
    history = data[lr]
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    gap = [t - v for t, v in zip(train_acc, val_acc)]
    
    ax.plot(history['epochs'], gap, 
            marker='o', label=f'LR={lr}', linewidth=2.5,
            color=colors[idx], markersize=6)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Train Acc - Val Acc (Generalization Gap)', fontsize=12)
ax.set_title('Smaller gap = Better generalization', fontsize=13)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 05_overfitting_analysis.png")
plt.close()

# ============================================================================
# 4. PRINT ANALYSIS SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("ANALYSIS SUMMARY - Learning Rate Effect on Training Behaviour")
print(f"{'='*70}\n")

print("📊 RESULTS TABLE:")
print(f"{'LR':>10} | {'Final Train':>12} | {'Final Val':>12} | {'Best Val':>11} | {'Best Epoch':>10}")
print(f"{'':>10} | {'Accuracy':>12} | {'Accuracy':>12} | {'Accuracy':>11} | {'':>10}")
print(f"{'-'*70}")

for row in summary:
    print(f"{row['learning_rate']:>10.4f} | {row['final_train_acc']:>12.4f} | "
          f"{row['final_val_acc']:>12.4f} | {row['best_val_acc']:>11.4f} | {row['best_val_epoch']:>10d}")

print(f"\n📈 KEY INSIGHTS:\n")

# Find best performing LR
best_idx = max(range(len(summary)), key=lambda i: summary[i]['best_val_acc'])
best_result = summary[best_idx]
print(f"✓ Best performing learning rate: {best_result['learning_rate']:.4f}")
print(f"  → Achieved best validation accuracy: {best_result['best_val_acc']:.4f}")
print(f"  → At epoch: {best_result['best_val_epoch']}")

# Find fastest convergence
fastest_idx = min(range(len(summary)), key=lambda i: summary[i]['best_val_epoch'])
fastest_result = summary[fastest_idx]
print(f"\n✓ Fastest convergence (to peak accuracy): LR = {fastest_result['learning_rate']:.4f}")
print(f"  → Reached peak at epoch: {fastest_result['best_val_epoch']}")

# Analyze stability
print(f"\n✓ Learning rate sensitivity:")
accs = [s['best_val_acc'] for s in summary]
acc_std = max(accs) - min(accs)
print(f"  → Accuracy range: {min(accs):.4f} - {max(accs):.4f}")
print(f"  → Variation: {acc_std:.4f}")

print(f"\n{'='*70}")
print(f"✅ All visualizations saved!")
print(f"   1. 01_loss_comparison.png - Loss curves for different LRs")
print(f"   2. 02_accuracy_comparison.png - Accuracy curves for different LRs")
print(f"   3. 03_final_performance.png - Bar charts of final performance")
print(f"   4. 04_convergence_speed.png - How fast each LR reaches target")
print(f"   5. 05_overfitting_analysis.png - Generalization gap analysis")
print(f"{'='*70}")
print(f"\n📝 Ready for thesis section 4.3!")
