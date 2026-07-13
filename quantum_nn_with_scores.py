#!/usr/bin/env python3
"""
Quantum Kernel-Enhanced Neural Network for DDoS Detection — WITH SCORE SAVING FOR ROC CURVES
Uses Qiskit ZZFeatureMap (same as QSVM) for quantum feature extraction
Classical MLP for classification
Saves: predict_proba scores per fold for ROC curve generation
Fast: kernel pre-computed once, classical NN trains in ~5 seconds
Ver: 2.0 | Date: 2026-07-10
"""

import numpy as np
import csv
import os
import sys
import warnings
import time

warnings.filterwarnings('ignore')

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler

DATASET_MAP = {
    'nsl-kdd': '/root/ddos-framework/data/standard_datasets/processed/nsl-kdd_processed.csv',
    'unsw-nb15': '/root/ddos-framework/data/standard_datasets/processed/unsw-nb15_processed.csv',
    'cicddos2019': '/root/ddos-framework/data/standard_datasets/processed/cicddos2019_processed.csv',
    'cicids2017': '/root/ddos-framework/data/standard_datasets/processed/cicids2017_processed.csv',
    'kddcup1999': '/root/ddos-framework/data/standard_datasets/processed/kddcup1999_processed.csv',
}

# Create directory for saving scores
SCORES_DIR = '/root/ddos-framework/results/scores/quantum_nn'
os.makedirs(SCORES_DIR, exist_ok=True)


def load_data(path):
    all_data, labels = [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features = [float(x) for x in row[:6]]
            label = int(row[6])
            all_data.append(features)
            labels.append(label)
    return np.array(all_data), np.array(labels)


def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def extract_quantum_kernel_features(X_train, X_test):
    """
    Use Qiskit ZZFeatureMap (same as QSVM) to compute quantum kernel matrix.
    Return kernel matrix features for classical NN.
    """
    feature_map = ZZFeatureMap(feature_dimension=6, reps=2, entanglement='circular')
    kernel = FidelityStatevectorKernel(feature_map=feature_map)

    # Compute kernel matrices
    K_train = kernel.evaluate(x_vec=X_train)
    K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)

    return K_train, K_test


def evaluate_dataset(name, path, k):
    X, y = load_data(path)

    # Scale to [0, pi] for quantum encoding (same as QSVM)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accs, precs, recs, f1s, specs, aucs = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
        print(f"  Fold {fold}/{k}...", flush=True)

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()

        # Step 1: Compute quantum kernel features (fast, using statevector)
        print(f"    Computing quantum kernel...", flush=True)
        K_train, K_test = extract_quantum_kernel_features(X_train, X_test)

        # Step 2: Scale kernel features
        k_scaler = MinMaxScaler(feature_range=(0, 1))
        K_train = k_scaler.fit_transform(K_train)
        K_test = k_scaler.transform(K_test)

        # Step 3: Train classical MLP on quantum kernel features
        print(f"    Training classical MLP...", flush=True)
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        clf.fit(K_train, y_train)

        train_time = time.time() - start

        # ========== SAVE SCORES FOR ROC CURVES ==========
        y_proba = clf.predict_proba(K_test)[:, 1]  # Probability of class 1 (attack)

        # Save scores and true labels for this fold
        score_file = f'{SCORES_DIR}/{name}_k{k}_fold{fold}_scores.npy'
        ytrue_file = f'{SCORES_DIR}/{name}_k{k}_fold{fold}_ytrue.npy'
        np.save(score_file, y_proba)
        np.save(ytrue_file, y_test)
        print(f"    Saved scores: {score_file}")
        # =================================================

        # Predict
        y_pred = clf.predict(K_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        specs.append(specificity_score(y_test, y_pred))

        # Compute AUC from predict_proba scores
        aucs.append(roc_auc_score(y_test, y_proba))

        print(f"    Training time: {train_time:.1f}s")

    return {
        'accuracy': np.mean(accs), 'precision': np.mean(precs),
        'recall': np.mean(recs), 'f1': np.mean(f1s),
        'specificity': np.mean(specs), 'auc': np.mean(aucs)
    }


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'nsl-kdd'
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    path = DATASET_MAP.get(dataset_name)

    if not path or not os.path.exists(path):
        print(f"ERROR: Dataset {dataset_name} not found")
        sys.exit(1)

    print(f"=== QUANTUM KERNEL NN {dataset_name} k={k} ===")
    print(f"Architecture: ZZFeatureMap (reps=2, circular) + Classical MLP(128->64)")
    print(f"Quantum: FidelityStatevectorKernel (same as QSVM)")
    print(f"Classical: Adam, 500 iterations")
    print(f"WITH SCORE SAVING FOR ROC CURVES")
    print()

    start = time.time()
    results = evaluate_dataset(dataset_name, path, k)

    print(f"\nAVERAGE ACROSS {k} FOLDS:")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  F1-Score:    {results['f1']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  AUC:         {results['auc']:.4f}")
    print(f"\nTotal time: {time.time() - start:.1f}s")