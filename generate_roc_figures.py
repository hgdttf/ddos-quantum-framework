#!/usr/bin/env python3
"""
Generate publication-quality ROC curves with PCHIP smoothing.
Uses real k-fold scores. PCHIP preserves monotonicity, no overshoot.
MUST disclose PCHIP in paper caption — this is a Q1 requirement.
Ver: 5.0 | Date: 2026-07-12
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

# HIGH-CONTRAST colors
COLORS = {
    'Classical SVM': '#D55E00',  # vermillion
    'Quantum SVM': '#009E73',  # bluish green
    'Classical NN': '#0072B2',  # blue
    'Quantum NN': '#CC79A7'  # reddish purple
}


def load_fold_scores(classifier, dataset, k, fold):
    score_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_scores.npy'
    ytrue_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_ytrue.npy'
    if not os.path.exists(score_file) or not os.path.exists(ytrue_file):
        return None, None
    return np.load(score_file), np.load(ytrue_file)


def smooth_roc_pchip(fpr, tpr, n_points=500):
    """
    Smooth ROC curve using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
    PCHIP guarantees:
    - Monotonicity (TPR never decreases)
    - No overshoot (no artificial peaks)
    - C1 continuity (smooth, no sharp corners)
    """
    if len(fpr) < 3:
        return fpr, tpr

    # Remove duplicate FPR values (PCHIP requires strictly increasing x)
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

    # PCHIP interpolation
    pchip = PchipInterpolator(fpr_clean, tpr_clean)
    fpr_smooth = np.linspace(0, 1, n_points)
    tpr_smooth = pchip(fpr_smooth)
    tpr_smooth = np.clip(tpr_smooth, 0, 1)

    # Ensure exact endpoints
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

    # Use median AUC fold as representative
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

        ax.plot(fpr, tpr,
                color=COLORS[clf_name],
                linestyle='-',
                linewidth=2.5,
                label=f'{clf_name} (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier (AUC = 0.5000)')

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

    # Title BELOW
    fig.text(0.5, 0.02, f'Receiver Operating Characteristic — {dataset_label}',
             ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    safe_name = dataset.lower().replace(' ', '_').replace('-', '_')
    filepath = f'{OUTPUT_DIR}/roc_{safe_name}_final.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    print("=" * 70)
    print("GENERATING ROC CURVES — PCHIP SMOOTHED, PUBLICATION-READY")
    print("=" * 70)
    print("REMEMBER: Disclose PCHIP in your paper caption!")
    print()

    for dataset in DATASETS:
        print(f"Processing {DATASET_LABELS[dataset]}...")
        generate_figure(dataset, DATASET_LABELS[dataset])

    print(f"\n{'=' * 70}")
    print("ALL FIGURES SAVED")
    print(f"{'=' * 70}")
    print("\nMANDATORY CAPTION ADDITION:")
    print('"ROC curves are smoothed using Piecewise Cubic Hermite Interpolating"')
    print('"Polynomial (PCHIP) interpolation to preserve monotonicity while"')
    print('"enhancing visual clarity. Raw empirical curves are available in"')
    print('"supplementary materials."')
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()