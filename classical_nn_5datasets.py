#!/usr/bin/env python3
"""
Classical Neural Network (MLP) for DDoS Detection
5 Datasets: NSL-KDD, UNSW-NB15, CIC-DDoS2019, CICIDS2017, KDD Cup 1999
k-fold CV: k=2,3,4,5
Metrics: Accuracy, Precision, Recall, F1-Score, Specificity
Architecture: 2 hidden layers (128 -> 64), ReLU, Adam
"""

import numpy as np
import csv
import os
import sys
import warnings

warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

DATASET_MAP = {
    'nsl-kdd': '/root/ddos-framework/data/standard_datasets/processed/nsl-kdd_processed.csv',
    'unsw-nb15': '/root/ddos-framework/data/standard_datasets/processed/unsw-nb15_processed.csv',
    'cicddos2019': '/root/ddos-framework/data/standard_datasets/processed/cicddos2019_processed.csv',
    'cicids2017': '/root/ddos-framework/data/standard_datasets/processed/cicids2017_processed.csv',
    'kddcup1999': '/root/ddos-framework/data/standard_datasets/processed/kddcup1999_processed.csv',
}


def load_data(path):
    """Load processed CSV: 6 features + 1 label."""
    all_data = []
    labels = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features = [float(x) for x in row[:6]]
            label = int(row[6])
            all_data.append(features)
            labels.append(label)
    return np.array(all_data), np.array(labels)


def specificity_score(y_true, y_pred):
    """Compute specificity: TN / (TN + FP)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_dataset(name, path, k):
    """Run k-fold CV and return average metrics."""
    X, y = load_data(path)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accs = []
    precs = []
    recs = []
    f1s = []
    specs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            early_stopping=False
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        specs.append(specificity_score(y_test, y_pred))

    return {
        'accuracy': np.mean(accs),
        'precision': np.mean(precs),
        'recall': np.mean(recs),
        'f1': np.mean(f1s),
        'specificity': np.mean(specs)
    }


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'nsl-kdd'
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    path = DATASET_MAP.get(dataset_name)
    if not path or not os.path.exists(path):
        print(f"ERROR: Dataset {dataset_name} not found at {path}")
        sys.exit(1)

    print(f"=== CLASSICAL NN {dataset_name} k={k} ===")
    print(f"Architecture: 2 hidden layers (128 -> 64), ReLU, Adam")
    print(f"Scaler: MinMax [0, 1]")
    print()

    results = evaluate_dataset(dataset_name, path, k)

    print(f"AVERAGE ACROSS {k} FOLDS:")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  F1-Score:    {results['f1']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")