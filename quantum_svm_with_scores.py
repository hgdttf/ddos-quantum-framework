#!/usr/bin/env python3
"""
Quantum SVM for 5 Standard Datasets — WITH SCORE SAVING FOR ROC CURVES
k-Fold Cross-Validation: k = 2, 3, 4, 5
Config: zz_feature_map, reps=2, CIRCULAR for most, FULL for cicids2017
Saves: decision_function scores per fold for ROC curve generation
Ver: 7.1 | Date: 2026-07-10
"""

import numpy as np
import csv
import os
import sys
import signal
import gc

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, f1_score, roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler

from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit.circuit.library import zz_feature_map

DATASET_MAP = {
    'nsl-kdd': '/root/ddos-framework/data/standard_datasets/processed/nsl-kdd_processed.csv',
    'unsw-nb15': '/root/ddos-framework/data/standard_datasets/processed/unsw-nb15_processed.csv',
    'cicddos2019': '/root/ddos-framework/data/standard_datasets/processed/cicddos2019_processed.csv',
    'cicids2017': '/root/ddos-framework/data/standard_datasets/processed/cicids2017_processed.csv',
    'kddcup1999': '/root/ddos-framework/data/standard_datasets/processed/kddcup1999_processed.csv',
}

SCORES_DIR = '/root/ddos-framework/results/scores/quantum_svm'
os.makedirs(SCORES_DIR, exist_ok=True)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Fold computation timed out")


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

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    # ORIGINAL CONFIG: circular for most, full only for cicids2017
    if dataset_name == 'cicids2017':
        entanglement = 'full'
        print(f"\nUsing FULL entanglement for {dataset_name}")
    else:
        entanglement = 'circular'
        print(f"\nUsing CIRCULAR entanglement for {dataset_name}")

    print(f"\n{'=' * 60}")
    print(f"QUANTUM SVM {k}-FOLD CROSS-VALIDATION ({dataset_name.upper()})")
    print(f"WITH SCORE SAVING FOR ROC CURVES")
    print(f"Config: zz_feature_map, reps=2, {entanglement.upper()}, FidelityStatevectorKernel")
    print(f"{'=' * 60}")

    feature_map = zz_feature_map(feature_dimension=6, reps=2, entanglement=entanglement)
    quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{k}: Starting...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)

        try:
            print(f"Fold {fold}/{k}: Computing kernel matrices...")
            K_train = quantum_kernel.evaluate(x_vec=X_train)
            K_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

            print(f"Fold {fold}/{k}: Training QSVM...")
            clf = SVC(kernel='precomputed', C=1.0, class_weight='balanced', max_iter=5000)
            clf.fit(K_train, y_train)

            # SAVE SCORES FOR ROC CURVES
            y_scores = clf.decision_function(K_test)
            score_file = f'{SCORES_DIR}/{dataset_name}_k{k}_fold{fold}_scores.npy'
            ytrue_file = f'{SCORES_DIR}/{dataset_name}_k{k}_fold{fold}_ytrue.npy'
            np.save(score_file, y_scores)
            np.save(ytrue_file, y_test)
            print(f"  Saved scores: {score_file}")

            y_pred = clf.predict(K_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_scores)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            print(f"  Accuracy:    {acc:.4f}")
            print(f"  Precision:   {prec:.4f}")
            print(f"  Recall:      {rec:.4f}")
            print(f"  F1-Score:    {f1:.4f}")
            print(f"  AUC:         {auc:.4f}")
            print(f"  Sensitivity: {sensitivity:.4f}")
            print(f"  Specificity: {specificity:.4f}")

            results.append((acc, prec, rec, f1, sensitivity, specificity, auc))

        except TimeoutException:
            print(f"  WARNING: Fold {fold}/{k} timed out after 10 minutes. Skipping.")
            results.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        except Exception as e:
            print(f"  ERROR in Fold {fold}/{k}: {str(e)}")
            results.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        finally:
            signal.alarm(0)

        gc.collect()

    valid_results = [r for r in results if r[0] > 0]

    if len(valid_results) == 0:
        print(f"\n{'=' * 60}")
        print(f"ALL FOLDS FAILED")
        print(f"{'=' * 60}")
        return None

    avg = [np.mean([r[i] for r in valid_results]) for i in range(7)]
    print(f"\n{'=' * 60}")
    print(f"AVERAGE ACROSS {len(valid_results)} VALID FOLDS:")
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
        print("Usage: python quantum_svm_with_scores.py <dataset_name> <k>")
        print("Datasets: nsl-kdd, unsw-nb15, cicddos2019, cicids2017, kddcup1999")
        sys.exit(1)

    dataset_name = sys.argv[1]
    k = int(sys.argv[2])
    run_kfold(dataset_name, k)