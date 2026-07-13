#!/usr/bin/env python3
"""
Classical SVM for 5 Standard Datasets — WITH SCORE SAVING FOR ROC CURVES
k-Fold Cross-Validation: k = 2, 3, 4, 5
Saves: decision_function scores per fold for ROC curve generation
ORIGINAL CONFIG: MinMaxScaler [0,1] only — NO QuantileTransformer
Ver: 3.1 | Date: 2026-07-10
"""

import numpy as np
import csv
import os
import sys

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, f1_score, roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler

DATASET_MAP = {
    'nsl-kdd': '/root/ddos-framework/data/standard_datasets/processed/nsl-kdd_processed.csv',
    'unsw-nb15': '/root/ddos-framework/data/standard_datasets/processed/unsw-nb15_processed.csv',
    'cicddos2019': '/root/ddos-framework/data/standard_datasets/processed/cicddos2019_processed.csv',
    'cicids2017': '/root/ddos-framework/data/standard_datasets/processed/cicids2017_processed.csv',
    'kddcup1999': '/root/ddos-framework/data/standard_datasets/processed/kddcup1999_processed.csv',
}

SCORES_DIR = '/root/ddos-framework/results/scores/classical_svm'
os.makedirs(SCORES_DIR, exist_ok=True)


def load_data(dataset_name):
    path = DATASET_MAP.get(dataset_name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {dataset_name} at {path}")

    all_data, labels = [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features = [float(x) for x in row[:6]]
            label = int(row[6])
            all_data.append(features)
            labels.append(label)
    return np.array(all_data), np.array(labels)


def run_kfold(dataset_name, k):
    X, y = load_data(dataset_name)
    print(f"Loaded {len(X)} vectors from {dataset_name}")
    print(f"Attack: {sum(y)}, Normal: {len(y) - sum(y)}")

    # ORIGINAL CONFIG: Only MinMaxScaler [0, 1] — NO QuantileTransformer
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    print(f"\n{'=' * 60}")
    print(f"CLASSICAL SVM {k}-FOLD CROSS-VALIDATION ({dataset_name.upper()})")
    print(f"WITH SCORE SAVING FOR ROC CURVES")
    print(f"Config: RBF kernel, C=1.0, MinMaxScaler [0,1] — NO QuantileTransformer")
    print(f"{'=' * 60}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', gamma='scale')
        clf.fit(X_train, y_train)

        # SAVE SCORES FOR ROC CURVES
        y_scores = clf.decision_function(X_test)
        score_file = f'{SCORES_DIR}/{dataset_name}_k{k}_fold{fold}_scores.npy'
        ytrue_file = f'{SCORES_DIR}/{dataset_name}_k{k}_fold{fold}_ytrue.npy'
        np.save(score_file, y_scores)
        np.save(ytrue_file, y_test)
        print(f"  Saved scores: {score_file}")

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_scores)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"\nFold {fold}/{k}: Training SVM...")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  AUC:         {auc:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")

        results.append((acc, prec, rec, f1, sensitivity, specificity, auc))

    avg = [np.mean([r[i] for r in results]) for i in range(7)]
    print(f"\n{'=' * 60}")
    print(f"AVERAGE ACROSS {k} FOLDS:")
    print(f"  Accuracy:    {avg[0]:.4f}")
    print(f"  Precision:   {avg[1]:.4f}")
    print(f"  Recall:      {avg[2]:.4f}")
    print(f"  F1-Score:    {avg[3]:.4f}")
    print(f"  Sensitivity: {avg[4]:.4f}")
    print(f"  Specificity: {avg[5]:.4f}")
    print(f"  AUC:         {avg[6]:.4f}")
    print(f"{'=' * 60}")

    return avg


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python classical_svm_with_scores.py <dataset> <k>")
        print("Datasets: nsl-kdd, unsw-nb15, cicddos2019, cicids2017, kddcup1999")
        print("k values: 2, 3, 4, 5")
        sys.exit(1)

    dataset_name = sys.argv[1]
    k = int(sys.argv[2])

    if dataset_name not in DATASET_MAP:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)

    if k not in [2, 3, 4, 5]:
        print(f"Invalid k: {k}. Use 2, 3, 4, or 5.")
        sys.exit(1)

    run_kfold(dataset_name, k)