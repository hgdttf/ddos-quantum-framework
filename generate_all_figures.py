#!/usr/bin/env python3
"""
Generate ALL Figures for Paper — CORRECTED VERSION
Uses published paper values for bar graphs.
Uses averaged ROC curves across k=2-5 folds with confidence bands.
Ver: 2.0 | Date: 2026-07-11
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import scipy.interpolate as interp

# ============================================================
# CONFIGURATION
# ============================================================

SCORES_BASE = '/root/ddos-framework/results/scores'
FIGURES_DIR = '/root/ddos-framework/results/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ['nsl-kdd', 'unsw-nb15', 'cicddos2019', 'cicids2017', 'kddcup1999']
DATASET_LABELS = ['NSL-KDD', 'UNSW-NB15', 'CIC-DDoS2019', 'CICIDS2017', 'KDD Cup 1999']

METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'specificity']

# Colorblind-safe palette
COLORS = {
    'classical_svm': '#1f77b4',  # blue
    'quantum_svm': '#ff7f0e',  # orange
    'classical_nn': '#2ca02c',  # green
    'quantum_nn': '#d62728',  # red
}

# ============================================================
# CORRECTED TABLE VALUES FROM YOUR PUBLISHED PAPER
# Table 11: Classical SVM vs Quantum SVM (averages across k=2,3,4,5)
# ============================================================

TABLE_11 = {
    'nsl-kdd': {
        'classical_svm': {'accuracy': 0.8885, 'precision': 0.9675, 'recall': 0.8040,
                          'f1': 0.8780, 'specificity': 0.9730},
        'quantum_svm': {'accuracy': 0.8483, 'precision': 0.9645, 'recall': 0.7231,
                        'f1': 0.8265, 'specificity': 0.9735},
    },
    'unsw-nb15': {
        'classical_svm': {'accuracy': 0.8080, 'precision': 0.7813, 'recall': 0.8861,
                          'f1': 0.8170, 'specificity': 0.7299},
        'quantum_svm': {'accuracy': 0.8556, 'precision': 0.8799, 'recall': 0.8236,
                        'f1': 0.8508, 'specificity': 0.8876},
    },
    'cicddos2019': {
        'classical_svm': {'accuracy': 0.8667, 'precision': 0.8391, 'recall': 0.9075,
                          'f1': 0.8719, 'specificity': 0.8260},
        'quantum_svm': {'accuracy': 0.9478, 'precision': 0.9123, 'recall': 0.9920,
                        'f1': 0.9503, 'specificity': 0.9035},
    },
    'kddcup1999': {
        'classical_svm': {'accuracy': 0.9730, 'precision': 1.0000, 'recall': 0.9460,
                          'f1': 0.9722, 'specificity': 1.0000},
        'quantum_svm': {'accuracy': 0.9860, 'precision': 1.0000, 'recall': 0.9720,
                        'f1': 0.9858, 'specificity': 1.0000},
    },
    'cicids2017': {
        'classical_svm': {'accuracy': 0.7157, 'precision': 0.7341, 'recall': 0.6762,
                          'f1': 0.7038, 'specificity': 0.7551},
        'quantum_svm': {'accuracy': 0.8242, 'precision': 0.8336, 'recall': 0.8125,
                        'f1': 0.8228, 'specificity': 0.8375},
    },
}

# ============================================================
# CORRECTED TABLE 22: Classical NN vs Quantum NN
# FIXED: CIC-DDoS2019 Quantum NN accuracy = 0.9388 (not 0.9390)
# ============================================================

TABLE_22 = {
    'nsl-kdd': {
        'classical_nn': {'accuracy': 0.8928, 'precision': 0.9711, 'recall': 0.8095,
                         'f1': 0.8828, 'specificity': 0.9760},
        'quantum_nn': {'accuracy': 0.8662, 'precision': 0.9125, 'recall': 0.8135,
                       'f1': 0.8586, 'specificity': 0.9190},
    },
    'unsw-nb15': {
        'classical_nn': {'accuracy': 0.8562, 'precision': 0.8359, 'recall': 0.8885,
                         'f1': 0.8607, 'specificity': 0.8240},
        'quantum_nn': {'accuracy': 0.8467, 'precision': 0.8491, 'recall': 0.8465,
                       'f1': 0.8465, 'specificity': 0.8470},
    },
    'cicddos2019': {
        'classical_nn': {'accuracy': 0.9225, 'precision': 0.9041, 'recall': 0.9460,
                         'f1': 0.9239, 'specificity': 0.8990},
        'quantum_nn': {'accuracy': 0.9388, 'precision': 0.9218, 'recall': 0.9610,
                       'f1': 0.9401, 'specificity': 0.9170},
    },
    'kddcup1999': {
        'classical_nn': {'accuracy': 0.9895, 'precision': 0.9985, 'recall': 0.9805,
                         'f1': 0.9894, 'specificity': 0.9985},
        'quantum_nn': {'accuracy': 0.9888, 'precision': 0.9940, 'recall': 0.9835,
                       'f1': 0.9887, 'specificity': 0.9940},
    },
    'cicids2017': {
        'classical_nn': {'accuracy': 0.8298, 'precision': 0.8474, 'recall': 0.8070,
                         'f1': 0.8252, 'specificity': 0.8525},
        'quantum_nn': {'accuracy': 0.8365, 'precision': 0.8507, 'recall': 0.8190,
                       'f1': 0.8334, 'specificity': 0.8540},
    },
}


# ============================================================
# FIGURE 8(a): Bar Graph — Classical SVM vs Quantum SVM (Table 11 style)
# Like Sir's Figure 13
# ============================================================

def generate_figure_svm_bars():
    """Grouped bar chart for SVM comparison across 5 datasets."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(DATASETS))
    width = 0.15

    for i, metric in enumerate(METRIC_KEYS):
        c_vals = [TABLE_11[d]['classical_svm'][metric] for d in DATASETS]
        q_vals = [TABLE_11[d]['quantum_svm'][metric] for d in DATASETS]

        offset = (i - 2) * width

        ax.bar(x + offset - width / 2, c_vals, width,
               label=f'Classical SVM — {METRICS[i]}' if i == 0 else "",
               color=COLORS['classical_svm'], alpha=0.6 + i * 0.08,
               edgecolor='black', linewidth=0.3)
        ax.bar(x + offset + width / 2, q_vals, width,
               label=f'QSVM — {METRICS[i]}' if i == 0 else "",
               color=COLORS['quantum_svm'], alpha=0.6 + i * 0.08,
               edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Classical SVM vs Quantum SVM\n'
                 'Across 5 Standard Datasets (Averaged over k=2,3,4,5 Folds)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure8a_svm_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/figure8a_svm_comparison.pdf', bbox_inches='tight')
    print(f"Saved: figure8a_svm_comparison.png")
    plt.close()


# ============================================================
# FIGURE 8(b): Bar Graph — Classical NN vs Quantum NN (Table 22 style)
# Like Sir's Figure 14
# ============================================================

def generate_figure_nn_bars():
    """Grouped bar chart for NN comparison across 5 datasets."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(DATASETS))
    width = 0.15

    for i, metric in enumerate(METRIC_KEYS):
        c_vals = [TABLE_22[d]['classical_nn'][metric] for d in DATASETS]
        q_vals = [TABLE_22[d]['quantum_nn'][metric] for d in DATASETS]

        offset = (i - 2) * width

        ax.bar(x + offset - width / 2, c_vals, width,
               label=f'Classical NN — {METRICS[i]}' if i == 0 else "",
               color=COLORS['classical_nn'], alpha=0.6 + i * 0.08,
               edgecolor='black', linewidth=0.3)
        ax.bar(x + offset + width / 2, q_vals, width,
               label=f'QNN — {METRICS[i]}' if i == 0 else "",
               color=COLORS['quantum_nn'], alpha=0.6 + i * 0.08,
               edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Classical NN vs Quantum NN\n'
                 'Across 5 Standard Datasets (Averaged over k=2,3,4,5 Folds)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure8b_nn_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/figure8b_nn_comparison.pdf', bbox_inches='tight')
    print(f"Saved: figure8b_nn_comparison.png")
    plt.close()


# ============================================================
# FIGURE 6(a-d): 4 Individual ROC Curves — AVERAGED across k=2-5
# Like Sir's Figure 15(a-d) but with confidence bands from CV
# ============================================================

def load_all_folds(classifier, dataset):
    """Load scores from all k values (2,3,4,5) and all folds."""
    all_fpr = []
    all_tpr = []
    all_auc = []

    for k in [2, 3, 4, 5]:
        for fold in range(1, k + 1):
            score_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_scores.npy'
            ytrue_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_ytrue.npy'

            if not os.path.exists(score_file):
                continue

            scores = np.load(score_file)
            y_true = np.load(ytrue_file)

            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)

            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_auc.append(auc)

    return all_fpr, all_tpr, all_auc


def average_roc_curves(all_fpr, all_tpr):
    """Average multiple ROC curves using interpolation."""
    # Create common FPR grid
    mean_fpr = np.linspace(0, 1, 100)

    interp_tprs = []
    for fpr, tpr in zip(all_fpr, all_tpr):
        interp_func = interp.interp1d(fpr, tpr, kind='linear',
                                      bounds_error=False, fill_value=(0, 1))
        interp_tprs.append(interp_func(mean_fpr))

    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)

    return mean_fpr, mean_tpr, std_tpr


def generate_individual_rocs_averaged():
    """
    4 individual ROC curves for CIC-DDoS2019.
    Each curve is AVERAGED across k=2,3,4,5 folds with std confidence band.
    """
    dataset = 'cicddos2019'

    classifiers = [
        ('classical_svm', 'Classical SVM (RBF kernel)', COLORS['classical_svm']),
        ('quantum_svm', 'Quantum SVM (ZZFeatureMap, FidelityKernel)', COLORS['quantum_svm']),
        ('classical_nn', 'Classical NN (MLP 128→64, ReLU)', COLORS['classical_nn']),
        ('quantum_nn', 'Quantum NN (ZZFeatureMap + MLP)', COLORS['quantum_nn']),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (clf_name, clf_label, color) in enumerate(classifiers):
        all_fpr, all_tpr, all_auc = load_all_folds(clf_name, dataset)

        if not all_auc:
            axes[idx].text(0.5, 0.5, 'Score files not found\nRun classifier first',
                           ha='center', va='center', fontsize=12, transform=axes[idx].transAxes)
            axes[idx].set_title(clf_label)
            continue

        mean_fpr, mean_tpr, std_tpr = average_roc_curves(all_fpr, all_tpr)
        mean_auc = np.mean(all_auc)
        std_auc = np.std(all_auc)

        # Plot mean ROC
        axes[idx].plot(mean_fpr, mean_tpr, color=color, linewidth=2.5,
                       label=f'{clf_label}\nAUC = {mean_auc:.4f} ± {std_auc:.4f}')

        # Plot confidence band (mean ± std)
        axes[idx].fill_between(mean_fpr,
                               np.maximum(mean_tpr - std_tpr, 0),
                               np.minimum(mean_tpr + std_tpr, 1),
                               color=color, alpha=0.2,
                               label=f'±1 std dev (n={len(all_auc)} folds)')

        # Random classifier line
        axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.50)')

        axes[idx].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
        axes[idx].set_ylabel('True Positive Rate (Recall)', fontsize=10)
        axes[idx].set_title(f'({chr(97 + idx)}) {clf_label}\n'
                            f'CIC-DDoS2019 (Averaged across k=2,3,4,5 folds)',
                            fontsize=11, fontweight='bold')
        axes[idx].legend(loc='lower right', fontsize=8)
        axes[idx].grid(alpha=0.3, linestyle='--')
        axes[idx].set_xlim([-0.02, 1.02])
        axes[idx].set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure6_individual_rocs.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/figure6_individual_rocs.pdf', bbox_inches='tight')
    print(f"Saved: figure6_individual_rocs.png")
    plt.close()


# ============================================================
# FIGURE 6(e): Combined ROC — All 4 Classifiers on One Plot
# Like Sir's Figure 15(e)
# ============================================================

def generate_combined_roc():
    """Combined ROC with all 4 classifiers, averaged across k=2-5."""
    dataset = 'cicddos2019'

    classifiers = [
        ('classical_svm', 'Classical SVM', COLORS['classical_svm']),
        ('quantum_svm', 'Quantum SVM', COLORS['quantum_svm']),
        ('classical_nn', 'Classical NN', COLORS['classical_nn']),
        ('quantum_nn', 'Quantum NN', COLORS['quantum_nn']),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    for clf_name, clf_label, color in classifiers:
        all_fpr, all_tpr, all_auc = load_all_folds(clf_name, dataset)

        if not all_auc:
            continue

        mean_fpr, mean_tpr, std_tpr = average_roc_curves(all_fpr, all_tpr)
        mean_auc = np.mean(all_auc)
        std_auc = np.std(all_auc)

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2.5,
                label=f'{clf_label} (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.50)')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax.set_title('Combined ROC Curves: All 4 Classifiers\n'
                 'CIC-DDoS2019 (Averaged across k=2,3,4,5 folds)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure6e_combined_roc.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/figure6e_combined_roc.pdf', bbox_inches='tight')
    print(f"Saved: figure6e_combined_roc.png")
    plt.close()


# ============================================================
# FIGURE 7: AUC Comparison Bar Chart
# Like Sir's Figure 16 but as bar chart (more informative)
# ============================================================

def load_auc_values():
    """Load computed AUC values from file or compute on the fly."""
    auc_data = {clf: {} for clf in ['classical_svm', 'quantum_svm',
                                    'classical_nn', 'quantum_nn']}

    for clf in auc_data.keys():
        for dataset in DATASETS:
            all_fpr, all_tpr, all_auc = load_all_folds(clf, dataset)
            if all_auc:
                auc_data[clf][dataset] = np.mean(all_auc)
            else:
                auc_data[clf][dataset] = 0.0

    return auc_data


def generate_auc_comparison():
    """Bar chart comparing AUC across all datasets and classifiers."""
    auc_data = load_auc_values()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(DATASETS))
    width = 0.2

    classifiers = [
        ('classical_svm', 'Classical SVM', COLORS['classical_svm']),
        ('quantum_svm', 'Quantum SVM', COLORS['quantum_svm']),
        ('classical_nn', 'Classical NN', COLORS['classical_nn']),
        ('quantum_nn', 'Quantum NN', COLORS['quantum_nn']),
    ]

    for idx, (clf_name, clf_label, color) in enumerate(classifiers):
        auc_vals = [auc_data[clf_name].get(d, 0) for d in DATASETS]

        bars = ax.bar(x + (idx - 1.5) * width, auc_vals, width,
                      label=clf_label, color=color, alpha=0.85,
                      edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, auc_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('AUC Comparison Across Datasets and Classifiers\n'
                 '(Computed from True ROC Curves, Averaged over k=2,3,4,5 Folds)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/figure7_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/figure7_auc_comparison.pdf', bbox_inches='tight')
    print(f"Saved: figure7_auc_comparison.png")
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING ALL FIGURES FOR PAPER — CORRECTED VERSION")
    print("=" * 70)

    # Figures that work NOW (no scores needed)
    print("\n[1/5] Generating SVM comparison bar graph (Figure 8a)...")
    generate_figure_svm_bars()

    print("\n[2/5] Generating NN comparison bar graph (Figure 8b)...")
    generate_figure_nn_bars()

    # Figures that require saved scores
    print("\n[3/5] Generating individual ROC curves (Figure 6a-d)...")
    print("      (Requires score files from all 4 classifiers)")
    generate_individual_rocs_averaged()

    print("\n[4/5] Generating combined ROC curve (Figure 6e)...")
    generate_combined_roc()

    print("\n[5/5] Generating AUC comparison bar chart (Figure 7)...")
    generate_auc_comparison()

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - figure8a_svm_comparison.png/pdf")
    print("  - figure8b_nn_comparison.png/pdf")
    print("  - figure6_individual_rocs.png/pdf")
    print("  - figure6e_combined_roc.png/pdf")
    print("  - figure7_auc_comparison.png/pdf")