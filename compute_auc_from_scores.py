#!/usr/bin/env python3
"""
Compute AUC from saved score files and generate table data.
Run AFTER running all 4 classifiers with score saving.
Ver: 1.0 | Date: 2026-07-11
"""

import numpy as np
import os
from sklearn.metrics import roc_auc_score

SCORES_BASE = '/root/ddos-framework/results/scores'
DATASETS = ['nsl-kdd', 'unsw-nb15', 'cicddos2019', 'cicids2017', 'kddcup1999']
DATASET_LABELS = ['NSL-KDD', 'UNSW-NB15', 'CIC-DDoS2019', 'CICIDS2017', 'KDD Cup 1999']
CLASSIFIERS = ['classical_svm', 'quantum_svm', 'classical_nn', 'quantum_nn']
K_VALUES = [2, 3, 4, 5]


def compute_auc_for_fold(classifier, dataset, k, fold):
    """Compute AUC for a single fold."""
    score_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_scores.npy'
    ytrue_file = f'{SCORES_BASE}/{classifier}/{dataset}_k{k}_fold{fold}_ytrue.npy'

    if not os.path.exists(score_file):
        return None

    scores = np.load(score_file)
    y_true = np.load(ytrue_file)

    return roc_auc_score(y_true, scores)


def compute_all_aucs():
    """Compute and print AUC table for all classifiers and datasets."""
    print("=" * 80)
    print("AUC COMPUTATION FROM SAVED SCORE FILES")
    print("=" * 80)

    results = {}

    for clf in CLASSIFIERS:
        results[clf] = {}
        print(f"\n{'=' * 60}")
        print(f"CLASSIFIER: {clf.upper()}")
        print(f"{'=' * 60}")

        for dataset, label in zip(DATASETS, DATASET_LABELS):
            aucs = []
            valid_k = []

            for k in K_VALUES:
                fold_aucs = []
                for fold in range(1, k + 1):
                    auc = compute_auc_for_fold(clf, dataset, k, fold)
                    if auc is not None:
                        fold_aucs.append(auc)

                if fold_aucs:
                    k_avg = np.mean(fold_aucs)
                    aucs.append(k_avg)
                    valid_k.append(k)
                    print(f"  {label} k={k}: AUC = {k_avg:.4f} (from {len(fold_aucs)} folds)")

            if aucs:
                overall_avg = np.mean(aucs)
                results[clf][dataset] = {
                    'aucs': aucs,
                    'overall': overall_avg,
                    'label': label
                }
                print(f"  >>> {label} OVERALL AVERAGE: {overall_avg:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("AUC SUMMARY TABLE (for paper)")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Classical SVM':<15} {'Quantum SVM':<15} {'Classical NN':<15} {'Quantum NN':<15}")
    print("-" * 80)

    for dataset, label in zip(DATASETS, DATASET_LABELS):
        row = f"{label:<15}"
        for clf in CLASSIFIERS:
            if dataset in results.get(clf, {}):
                row += f" {results[clf][dataset]['overall']:.4f}        "
            else:
                row += f" N/A           "
        print(row)

    # Save to file
    output_file = f'{SCORES_BASE}/computed_aucs.txt'
    with open(output_file, 'w') as f:
        f.write("AUC VALUES COMPUTED FROM SAVED SCORES\n")
        f.write("=" * 80 + "\n\n")
        for clf in CLASSIFIERS:
            f.write(f"\n{clf.upper()}\n")
            f.write("-" * 40 + "\n")
            for dataset in DATASETS:
                if dataset in results.get(clf, {}):
                    f.write(f"{dataset}: {results[clf][dataset]['overall']:.4f}\n")

    print(f"\nSaved to: {output_file}")
    return results


if __name__ == "__main__":
    compute_all_aucs()