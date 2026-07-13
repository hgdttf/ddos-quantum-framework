#!/usr/bin/env python3
"""
================================================================================
QUANTUM AMPLITUDE ESTIMATION (QAE) ANOMALY DETECTION
Full Evaluation on 5 Standard Datasets - Qiskit Aer Simulator Only
For: Quantum-based Multi-Vector DDoS Attack Detection Framework

HARDWARE: Qiskit Aer Simulator (local)
NOTE: Real quantum hardware validation performed separately on Open Quantum
      IonQ Forte-1 trapped-ion QPU (see qae_openquantum_validation.py)
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, StatePreparation
from qiskit_aer import AerSimulator

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "/root/ddos-framework/data/standard_datasets/processed"
OUTPUT_DIR = "/root/ddos-framework/data/qae_results"
DATASETS = {
    'NSL-KDD': 'nsl-kdd_processed.csv',
    'UNSW-NB15': 'unsw-nb15_processed.csv',
    'CIC-DDoS2019': 'cicddos2019_processed.csv',
    'KDD-Cup-1999': 'kddcup1999_processed.csv',
    'CICIDS2017': 'cicids2017_processed.csv'
}

N_FEATURES = 6
N_STATE_QUBITS = 3
N_EVAL_QUBITS = 3
THRESHOLD = 0.5
SHOTS = 8192
RANDOM_STATE = 42
K_FOLDS = [2, 3, 4, 5]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_dataset(name, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None, None

    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, :N_FEATURES].values
    y = df.iloc[:, N_FEATURES].values

    print(f"  Loaded {name}: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# =============================================================================
# STEP 2: NORMALIZATION
# =============================================================================

def normalize_features(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# STEP 3: AMPLITUDE ENCODING
# =============================================================================

def amplitude_encode(features):
    """
    Encode 6 features into 3-qubit quantum state amplitudes.
    Pad with 2 zeros to make 8 amplitudes (2^3).
    Normalize to unit vector.
    """
    padded = np.zeros(2 ** N_STATE_QUBITS)
    padded[:N_FEATURES] = features
    norm = np.linalg.norm(padded)
    if norm > 1e-10:
        padded = padded / norm
    else:
        padded[0] = 1.0
    return padded


# =============================================================================
# STEP 4: QAE CIRCUIT
# =============================================================================

def build_oracle(n_state_qubits):
    oracle = QuantumCircuit(n_state_qubits, name='Oracle')
    oracle.x(range(n_state_qubits))
    oracle.h(n_state_qubits - 1)
    if n_state_qubits == 3:
        oracle.ccx(0, 1, 2)
    else:
        oracle.mcx(list(range(n_state_qubits - 1)), n_state_qubits - 1)
    oracle.h(n_state_qubits - 1)
    oracle.x(range(n_state_qubits))
    return oracle


def build_diffuser(n_state_qubits):
    diffuser = QuantumCircuit(n_state_qubits, name='Diffuser')
    diffuser.h(range(n_state_qubits))
    diffuser.x(range(n_state_qubits))
    diffuser.h(n_state_qubits - 1)
    if n_state_qubits == 3:
        diffuser.ccx(0, 1, 2)
    else:
        diffuser.mcx(list(range(n_state_qubits - 1)), n_state_qubits - 1)
    diffuser.h(n_state_qubits - 1)
    diffuser.x(range(n_state_qubits))
    diffuser.h(range(n_state_qubits))
    return diffuser


def build_qae_circuit(encoded_features, n_eval_qubits=3):
    n_state = N_STATE_QUBITS

    eval_reg = QuantumRegister(n_eval_qubits, 'eval')
    state_reg = QuantumRegister(n_state, 'state')
    c_reg = ClassicalRegister(n_eval_qubits, 'c')
    qc = QuantumCircuit(eval_reg, state_reg, c_reg)

    # Use StatePreparation (Aer-compatible when transpiled)
    prep = StatePreparation(encoded_features)
    qc.compose(prep, qubits=state_reg, inplace=True)

    # Prepare evaluation qubits
    qc.h(eval_reg)

    # Build Grover operator
    oracle = build_oracle(n_state)
    diffuser = build_diffuser(n_state)

    # Controlled Grover iterations
    for j in range(n_eval_qubits):
        repetitions = 2 ** j
        for _ in range(repetitions):
            qc.compose(oracle, qubits=state_reg, inplace=True)
            qc.compose(diffuser, qubits=state_reg, inplace=True)

    # Inverse QFT
    qft_inv = QFT(n_eval_qubits, inverse=True)
    qc.compose(qft_inv, qubits=eval_reg, inplace=True)

    # Measure
    qc.measure(eval_reg, c_reg)
    return qc


# =============================================================================
# STEP 5: QAE PREDICTION
# =============================================================================

def run_qae_single_sample(feature_vector, shots=8192):
    encoded = amplitude_encode(feature_vector)
    qc = build_qae_circuit(encoded, n_eval_qubits=N_EVAL_QUBITS)

    # CRITICAL FIX: Transpile before running on AerSimulator
    # This converts StatePreparation and all gates to Aer-compatible basis gates
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)

    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    max_state = max(counts, key=counts.get)
    measured_int = int(max_state, 2)
    theta = np.pi * measured_int / (2 ** N_EVAL_QUBITS)
    prob_anomaly = np.sin(theta) ** 2
    prediction = 1 if prob_anomaly > THRESHOLD else 0

    return prediction, prob_anomaly


# =============================================================================
# STEP 6: EVALUATION
# =============================================================================

def evaluate_qae_fold(X_train, y_train, X_test, y_test, fold_num, dataset_name):
    print(f"    Fold {fold_num}: train={len(X_train)}, test={len(X_test)}")

    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    predictions = []
    probabilities = []
    start_time = time.time()

    for i, x in enumerate(X_test_scaled):
        pred, prob = run_qae_single_sample(x, shots=SHOTS)
        predictions.append(pred)
        probabilities.append(prob)

        if (i + 1) % 25 == 0 or i == len(X_test_scaled) - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(X_test_scaled) - i - 1)
            print(f"      {i + 1}/{len(X_test_scaled)} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    cm = confusion_matrix(y_test, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        tn = fp = fn = tp = 0
        specificity = 0

    print(f"    Fold {fold_num}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} Spec={specificity:.4f}")

    return {
        'fold': fold_num, 'k': None, 'dataset': dataset_name,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'specificity': specificity,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'predictions': predictions, 'probabilities': probabilities,
        'y_true': y_test
    }


def evaluate_dataset(name, filename):
    print(f"\n{'=' * 70}")
    print(f"DATASET: {name}")
    print(f"{'=' * 70}")

    X, y = load_dataset(name, filename)
    if X is None:
        return None

    fold_results = []

    for k in K_FOLDS:
        print(f"\n  --- k={k} fold cross-validation ---")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

        fold_idx = 1
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            result = evaluate_qae_fold(X_train, y_train, X_test, y_test, fold_idx, name)
            result['k'] = k
            fold_results.append(result)
            fold_idx += 1

    return fold_results


# =============================================================================
# STEP 7: RESULTS AGGREGATION
# =============================================================================

def aggregate_results(all_results):
    fold_data = []
    for dataset_results in all_results:
        if dataset_results is None:
            continue
        for fold in dataset_results:
            fold_data.append({
                'Dataset': fold['dataset'],
                'K (Folds)': fold['k'],
                'Fold': fold['fold'],
                'Accuracy': round(fold['accuracy'], 4),
                'Precision': round(fold['precision'], 4),
                'Recall': round(fold['recall'], 4),
                'F1-Score': round(fold['f1_score'], 4),
                'Specificity': round(fold['specificity'], 4)
            })
    df_folds = pd.DataFrame(fold_data)

    avg_data = []
    for dataset_results in all_results:
        if dataset_results is None:
            continue
        dataset_name = dataset_results[0]['dataset']
        for k in K_FOLDS:
            k_folds = [f for f in dataset_results if f['k'] == k]
            if not k_folds:
                continue
            avg_data.append({
                'Dataset': dataset_name,
                'K (Folds)': k,
                'Accuracy': round(np.mean([f['accuracy'] for f in k_folds]), 4),
                'Precision': round(np.mean([f['precision'] for f in k_folds]), 4),
                'Recall': round(np.mean([f['recall'] for f in k_folds]), 4),
                'F1-Score': round(np.mean([f['f1_score'] for f in k_folds]), 4),
                'Specificity': round(np.mean([f['specificity'] for f in k_folds]), 4)
            })
    df_avg = pd.DataFrame(avg_data)

    overall_data = []
    for dataset_results in all_results:
        if dataset_results is None:
            continue
        dataset_name = dataset_results[0]['dataset']
        overall_data.append({
            'Dataset': dataset_name,
            'Approach': 'QAE',
            'Avg Accuracy': round(np.mean([f['accuracy'] for f in dataset_results]), 4),
            'Avg Precision': round(np.mean([f['precision'] for f in dataset_results]), 4),
            'Avg Recall': round(np.mean([f['recall'] for f in dataset_results]), 4),
            'Avg F1-Score': round(np.mean([f['f1_score'] for f in dataset_results]), 4),
            'Avg Specificity': round(np.mean([f['specificity'] for f in dataset_results]), 4)
        })
    df_overall = pd.DataFrame(overall_data)

    return df_folds, df_avg, df_overall


def generate_confusion_matrices(all_results):
    cm_data = []
    for dataset_results in all_results:
        if dataset_results is None:
            continue
        dataset_name = dataset_results[0]['dataset']
        k2_folds = [f for f in dataset_results if f['k'] == 2]
        for fold in k2_folds:
            cm_data.append({
                'Dataset': dataset_name,
                'K': 2,
                'Fold': fold['fold'],
                'TN': fold['tn'],
                'FP': fold['fp'],
                'FN': fold['fn'],
                'TP': fold['tp'],
                'Total': len(fold['y_true'])
            })
    return pd.DataFrame(cm_data)


def save_results(df_folds, df_avg, df_overall, df_cm):
    df_folds.to_csv(os.path.join(OUTPUT_DIR, 'qae_per_fold_results.csv'), index=False)
    df_avg.to_csv(os.path.join(OUTPUT_DIR, 'qae_average_per_k.csv'), index=False)
    df_overall.to_csv(os.path.join(OUTPUT_DIR, 'qae_overall_summary.csv'), index=False)
    df_cm.to_csv(os.path.join(OUTPUT_DIR, 'qae_confusion_matrices_k2.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'qae_results_for_paper.txt'), 'w') as f:
        f.write("QAE ANOMALY DETECTION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("Table: Average Performance Across K-Folds\n")
        f.write("-" * 70 + "\n")
        f.write(df_avg.to_string(index=False))
        f.write("\n\nTable: Overall Average (All Folds)\n")
        f.write("-" * 70 + "\n")
        f.write(df_overall.to_string(index=False))
        f.write("\n\nTable: Confusion Matrices (k=2)\n")
        f.write("-" * 70 + "\n")
        f.write(df_cm.to_string(index=False))
        f.write("\n")

    print(f"\nResults saved to {OUTPUT_DIR}:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("QUANTUM AMPLITUDE ESTIMATION (QAE) ANOMALY DETECTION")
    print("Qiskit Aer Simulator Only")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  State Qubits: {N_STATE_QUBITS}")
    print(f"  Evaluation Qubits: {N_EVAL_QUBITS}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Shots: {SHOTS}")
    print(f"  Hardware: Qiskit Aer Simulator (local)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\nNOTE: Real hardware validation on Open Quantum IonQ Forte-1")
    print(f"      performed separately (see qae_openquantum_validation.py)\n")

    all_results = []
    for name, filename in DATASETS.items():
        results = evaluate_dataset(name, filename)
        if results is not None:
            all_results.append(results)

    if not all_results:
        print("\nERROR: No datasets evaluated.")
        return

    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    df_folds, df_avg, df_overall = aggregate_results(all_results)
    df_cm = generate_confusion_matrices(all_results)

    print("\n--- Per-Fold Results ---")
    print(df_folds.to_string(index=False))
    print("\n--- Average Per K ---")
    print(df_avg.to_string(index=False))
    print("\n--- Overall Summary ---")
    print(df_overall.to_string(index=False))
    print("\n--- Confusion Matrices (k=2) ---")
    print(df_cm.to_string(index=False))

    save_results(df_folds, df_avg, df_overall, df_cm)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()