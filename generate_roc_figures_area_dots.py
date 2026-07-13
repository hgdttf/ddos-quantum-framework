#!/usr/bin/env python3
"""
Generate PREMIUM publication-quality ROC curves with PCHIP smoothing.
Features: VISIBLE colored dots filling area UNDER each curve (same color as line).
Ver: 9.0 | Date: 2026-07-13
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import PchipInterpolator
import os

SCORES_BASE = '/root/ddos-framework/results/scores'
OUTPUT_DIR = '/root/ddos-framework/results/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ['nsl-kdd', 'unsw-nb15', 'cicddos2019', 'cicids2017', 'kddcup1999']
DATASET_LABELS = {
    'nsl-kdd': 'NSL-KDD',
    'unsw-nb15': 'UNSW-NB15',
    'cicddos2019': 'CIC-DDoS2019',
    'cicids2017': 'CICIDS2017',
    'kddcup1999': 'KDD Cup 1999'
}

CLASSIFIERS = {
    'classical_svm': 'Classical SVM',
    'quantum_svm': 'Quantum SVM',
    'classical_nn': 'Classical NN',
    'quantum_nn': 'Quantum NN'
}

# HIGH-CONTRAST, VIBRANT colors
COLORS = {
    'Classical SVM': '#E74C3C',  # vibrant red
    'Quantum SVM': '#2ECC71',  # vibrant green
    'Classical NN': '#3498DB',  # vibrant blue
    'Quantum NN': '#9B59B6'  # vibrant purple
}


def load_fold_scores(classifier, dataset, k, fold):
    score_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_scores.npy'
    ytrue_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_ytrue.npy'
    if not os.path.exists(score_file) or not os.path.exists(ytrue_file):
        return None, None
    return np.load(score_file), np.load(ytrue_file)


def smooth_roc_pchip(fpr, tpr, n_points=500):
    """Smooth ROC curve using PCHIP interpolation."""
    if len(fpr) < 3:
        return fpr, tpr

    fpr_clean = []
    tpr_clean = []
    last_fpr = -1
    for i in range(len(fpr)):
        if fpr[i] > last_fpr:
            fpr_clean.append(fpr[i])
            tpr_clean.append(tpr[i])
            last_fpr = fpr[i]

    fpr_clean = np.array(fpr_clean)
    tpr_clean = np.array(tpr_clean)

    if len(fpr_clean) < 3:
        return fpr, tpr

    pchip = PchipInterpolator(fpr_clean, tpr_clean)
    fpr_smooth = np.linspace(0, 1, n_points)
    tpr_smooth = pchip(fpr_smooth)
    tpr_smooth = np.clip(tpr_smooth, 0, 1)
    tpr_smooth[0] = 0.0
    tpr_smooth[-1] = 1.0

    return fpr_smooth, tpr_smooth


def compute_mean_roc(classifier, dataset, k):
    all_aucs = []
    all_fpr = []
    all_tpr = []

    for fold in range(1, k + 1):
        scores, y_true = load_fold_scores(classifier, dataset, k, fold)
        if scores is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_aucs.append(auc(fpr, tpr))

    if not all_fpr:
        return None, None, None, None

    median_idx = np.argsort(all_aucs)[len(all_aucs) // 2]
    fpr_smooth, tpr_smooth = smooth_roc_pchip(all_fpr[median_idx], all_tpr[median_idx])

    return fpr_smooth, tpr_smooth, np.mean(all_aucs), np.std(all_aucs)


def generate_figure(dataset, dataset_label):
    fig, ax = plt.subplots(figsize=(8, 8))
    k = 5

    for clf_key, clf_name in CLASSIFIERS.items():
        fpr, tpr, mean_auc, std_auc = compute_mean_roc(clf_key, dataset, k)
        if fpr is None:
            continue

        color = COLORS[clf_name]

        # Generate VISIBLE DOTS under the curve
        np.random.seed(hash(clf_name + dataset) % 10000)
        n_dots = 600  # Fewer dots, but visible

        # Sample x coordinates
        x_dots = np.random.uniform(0.01, 0.99, n_dots)
        x_dots = np.sort(x_dots)

        # Get curve height at each x
        pchip = PchipInterpolator(fpr, tpr)
        y_curve = pchip(x_dots)
        y_curve = np.clip(y_curve, x_dots, 1)

        # Sample y between diagonal and curve
        y_dots = np.random.uniform(x_dots, y_curve)

        # Plot VISIBLE dots — same color as line, higher alpha, larger size
        ax.scatter(x_dots, y_dots,
                   c=color,
                   s=8,  # BIGGER dots
                   alpha=0.35,  # MORE visible
                   edgecolors='none',
                   zorder=1)

        # Plot smooth line on top
        ax.plot(fpr, tpr,
                color=color,
                linestyle='-',
                linewidth=2.5,
                label=f'{clf_name} (AUC = {mean_auc:.4f} ± {std_auc:.4f})',
                zorder=10)

    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier (AUC = 0.5000)', zorder=5)

    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_title('')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.text(0.5, 0.02, f'Receiver Operating Characteristic — {dataset_label}',
             ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    safe_name = dataset.lower().replace(' ', '_').replace('-', '_')
    filepath = f'{OUTPUT_DIR}/roc_{safe_name}_visible_dots.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    print("=" * 70)
    print("GENERATING ROC CURVES — VISIBLE COLORED DOTS UNDER CURVE")
    print("=" * 70)

    for dataset in DATASETS:
        print(f"Processing {DATASET_LABELS[dataset]}...")
        generate_figure(dataset, DATASET_LABELS[dataset])

    print(f"\n{'=' * 70}")
    print("ALL FIGURES SAVED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()