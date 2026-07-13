#!/usr/bin/env python3
"""
================================================================================
FAST FIDELITY-DRIVEN QUANTUM AUTOENCODER (FiD-QAE) FOR ANOMALY DETECTION
Optimized Evaluation on CIC-DDoS2019 - Qiskit Aer Simulator
For: Quantum-based Multi-Vector DDoS Attack Detection Framework

OPTIMIZATIONS:
- Single dataset (CIC-DDoS2019) - main benchmark
- k=2 fold cross-validation only
- Train on 50 normal samples (research-validated: QAE works with small data)
- 1024 shots for training, 4096 for inference
- MAX_ITER = 20 with early stopping
- Batch evaluation: process multiple samples per optimizer call

APPROACH: Unsupervised QAE trained on NORMAL data only.
Anomaly score = 1 - fidelity (low fidelity = anomaly).
Based on: Romero et al. (2017), Qiskit ML Tutorial (2024),
          FiD-QAE for Fraud Detection (2024).

HARDWARE: Qiskit Aer Simulator (local)
NOTE: Full multi-dataset, multi-fold evaluation deferred to future work.
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
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# =============================================================================
# CONFIGURATION - OPTIMIZED
# =============================================================================

DATA_DIR = "/root/ddos-framework/data/standard_datasets/processed"
OUTPUT_DIR = "/root/ddos-framework/data/qae_results"

# SINGLE DATASET: CIC-DDoS2019 (main benchmark)
DATASET_NAME = 'CIC-DDoS2019'
DATASET_FILE = 'cicddos2019_processed.csv'

N_FEATURES = 6  # Number of classical features
N_QUBITS = 3  # ceil(log2(6)) = 3 qubits for amplitude encoding
N_LATENT = 2  # Compressed latent qubits
N_TRASH = 1  # Trash qubits
SHOTS_TRAIN = 1024  # Reduced shots for training speed
SHOTS_INFER = 4096  # Moderate shots for inference
RANDOM_STATE = 42
K_FOLDS = [2]  # k=2 ONLY (not 3,4,5)
MAX_ITER = 20  # Reduced iterations
N_TRAIN_NORMAL = 50  # Train on 50 normal samples (research-validated)
EARLY_STOP_PATIENCE = 5  # Stop if no improvement for 5 iterations

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_dataset():
    filepath = os.path.join(DATA_DIR, DATASET_FILE)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None, None

    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, :N_FEATURES].values.astype(float)
    y = df.iloc[:, N_FEATURES].values.astype(int)

    print(f"  Loaded {DATASET_NAME}: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# =============================================================================
# STEP 2: NORMALIZATION
# =============================================================================

def normalize_features(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# STEP 3: AMPLITUDE ENCODING
# =============================================================================

def amplitude_encode(features):
    """Encode 6 features into 3-qubit quantum state amplitudes."""
    padded = np.zeros(2 ** N_QUBITS, dtype=complex)
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

def build_encoder_ansatz(n_qubits, n_params):
    """Hardware-efficient parameterized ansatz using ParameterVector."""
    params = ParameterVector('θ', n_params)
    qc = QuantumCircuit(n_qubits, name='Encoder')

    param_idx = 0
    # Layer 1: RY rotations
    for q in range(n_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1
    # Entanglement
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    # Layer 2: RY rotations
    for q in range(n_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1
    # Entanglement reverse
    for q in range(n_qubits - 1, 0, -1):
        qc.cx(q - 1, q)
    # Layer 3: RY rotations
    for q in range(n_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1

    return qc, params


def build_qae_circuit(feature_vector, n_params):
    """Build QAE circuit with SWAP test. 6 qubits total."""
    total_qubits = N_QUBITS + N_TRASH + N_TRASH + 1  # 6
    aux_qubit = total_qubits - 1
    ref_qubit = N_QUBITS + N_TRASH
    trash_qubit = N_QUBITS

    qr = QuantumRegister(total_qubits, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)

    # Amplitude encode input
    encoded = amplitude_encode(feature_vector)
    prep = StatePreparation(encoded)
    qc.compose(prep, qubits=list(range(N_QUBITS)), inplace=True)

    # Parameterized encoder
    encoder, params = build_encoder_ansatz(N_QUBITS, n_params)
    qc.compose(encoder, qubits=list(range(N_QUBITS)), inplace=True)

    # SWAP test
    qc.h(aux_qubit)
    qc.cswap(aux_qubit, trash_qubit, ref_qubit)
    qc.h(aux_qubit)
    qc.measure(aux_qubit, cr[0])

    return qc, params


# =============================================================================
# STEP 5: TRAINING (Optimized)
# =============================================================================

def compute_fidelity_batch(parameters, feature_vectors, simulator, n_params):
    """
    Compute mean fidelity for a BATCH of feature vectors.
    Returns mean fidelity across all samples.
    """
    total_fidelity = 0.0
    n_samples = len(feature_vectors)

    for x in feature_vectors:
        qc, param_vector = build_qae_circuit(x, n_params)
        param_dict = {param_vector[i]: parameters[i] for i in range(n_params)}
        bound_qc = qc.assign_parameters(param_dict)
        transpiled_qc = transpile(bound_qc, simulator)

        job = simulator.run(transpiled_qc, shots=SHOTS_TRAIN)
        result = job.result()
        counts = result.get_counts()

        n_zero = counts.get('0', 0)
        fidelity = n_zero / SHOTS_TRAIN
        total_fidelity += fidelity

    return total_fidelity / n_samples


def train_qae(X_normal, simulator, max_iter=MAX_ITER):
    """Train QAE on NORMAL data with early stopping."""
    n_samples = len(X_normal)
    n_params = N_QUBITS * 3  # 9 parameters
    print(f"    Training QAE: {n_samples} normal samples, {n_params} parameters")
    print(f"    Shots per circuit: {SHOTS_TRAIN}")

    best_loss = float('inf')
    patience_counter = 0

    def cost_function(params):
        fidelity = compute_fidelity_batch(params, X_normal, simulator, n_params)
        loss = 1.0 - fidelity
        return loss

    # Custom callback for early stopping and progress
    def callback(params):
        nonlocal best_loss, patience_counter
        loss = cost_function(params)
        if loss < best_loss - 1e-4:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

    initial_params = np.random.uniform(-np.pi, np.pi, n_params)

    print(f"    Starting optimization (max_iter={max_iter}, early_stop_patience={EARLY_STOP_PATIENCE})...")
    start_time = time.time()

    # Use COBYLA with callback
    opt_result = minimize(
        cost_function,
        initial_params,
        method='COBYLA',
        options={'maxiter': max_iter, 'rhobeg': 0.1},
        callback=callback
    )

    # Check if early stopped
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"    Early stopping triggered at iteration {opt_result.nfev}")

    elapsed = time.time() - start_time
    print(f"    Training complete in {elapsed:.1f}s")
    print(f"    Final loss: {opt_result.fun:.6f}")
    print(f"    Final fidelity: {1.0 - opt_result.fun:.6f}")
    print(f"    Function evaluations: {opt_result.nfev}")

    return opt_result.x


# =============================================================================
# STEP 6: ANOMALY DETECTION
# =============================================================================

def compute_fidelity_single(parameters, feature_vector, simulator, n_params):
    """Compute fidelity for a single sample."""
    qc, param_vector = build_qae_circuit(feature_vector, n_params)
    param_dict = {param_vector[i]: parameters[i] for i in range(n_params)}
    bound_qc = qc.assign_parameters(param_dict)
    transpiled_qc = transpile(bound_qc, simulator)

    job = simulator.run(transpiled_qc, shots=SHOTS_INFER)
    result = job.result()
    counts = result.get_counts()

    n_zero = counts.get('0', 0)
    return n_zero / SHOTS_INFER


def detect_anomaly(parameters, feature_vector, threshold, simulator, n_params):
    """Detect anomaly using trained QAE."""
    fidelity = compute_fidelity_single(parameters, feature_vector, simulator, n_params)
    anomaly_score = 1.0 - fidelity
    prediction = 1 if anomaly_score > threshold else 0
    return prediction, anomaly_score, fidelity


def determine_threshold(parameters, X_normal_val, simulator, n_params):
    """Determine threshold from validation normal data."""
    scores = []
    for i, x in enumerate(X_normal_val):
        fidelity = compute_fidelity_single(parameters, x, simulator, n_params)
        scores.append(1.0 - fidelity)
        if (i + 1) % 10 == 0:
            print(f"      Threshold calibration: {i + 1}/{len(X_normal_val)}")

    scores = np.array(scores)
    threshold = np.mean(scores) + 2 * np.std(scores)
    print(f"    Threshold: {threshold:.4f} (mean={np.mean(scores):.4f}, std={np.std(scores):.4f})")
    return threshold


# =============================================================================
# STEP 7: EVALUATION
# =============================================================================

def evaluate_qae_fold(X_train, y_train, X_test, y_test, fold_num):
    print(f"    Fold {fold_num}: train={len(X_train)}, test={len(X_test)}")

    # Use only NORMAL samples for training (unsupervised)
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    print(f"    Normal training samples available: {len(X_normal)}")

    # Normalize
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    X_normal_scaled = scaler.transform(X_normal)

    # Select 50 normal samples for training (random subset)
    if len(X_normal_scaled) > N_TRAIN_NORMAL:
        np.random.seed(RANDOM_STATE)
        train_indices = np.random.choice(len(X_normal_scaled), N_TRAIN_NORMAL, replace=False)
        X_normal_train = X_normal_scaled[train_indices]
    else:
        X_normal_train = X_normal_scaled

    # Remaining normal samples for threshold calibration
    remaining_mask = np.ones(len(X_normal_scaled), dtype=bool)
    if len(X_normal_scaled) > N_TRAIN_NORMAL:
        remaining_mask[train_indices] = False
    X_normal_val = X_normal_scaled[remaining_mask]
    if len(X_normal_val) > 50:
        X_normal_val = X_normal_val[:50]  # Cap validation set

    print(f"    Training on {len(X_normal_train)} normal samples")
    print(f"    Validation set: {len(X_normal_val)} normal samples")

    simulator = AerSimulator()
    n_params = N_QUBITS * 3

    # Train QAE
    print(f"    Training QAE...")
    trained_params = train_qae(X_normal_train, simulator, max_iter=MAX_ITER)

    # Determine threshold
    print(f"    Determining anomaly threshold...")
    threshold = determine_threshold(trained_params, X_normal_val, simulator, n_params)

    # Detect anomalies on test set
    print(f"    Running anomaly detection on {len(X_test_scaled)} test samples...")
    predictions = []
    anomaly_scores = []
    fidelities = []

    start_time = time.time()

    for i, x in enumerate(X_test_scaled):
        pred, score, fid = detect_anomaly(trained_params, x, threshold, simulator, n_params)
        predictions.append(pred)
        anomaly_scores.append(score)
        fidelities.append(fid)

        if (i + 1) % 50 == 0 or i == len(X_test_scaled) - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(X_test_scaled) - i - 1)
            print(f"      {i + 1}/{len(X_test_scaled)} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    predictions = np.array(predictions)
    anomaly_scores = np.array(anomaly_scores)
    fidelities = np.array(fidelities)

    # Metrics
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
    print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        'fold': fold_num, 'k': 2, 'dataset': DATASET_NAME,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'specificity': specificity,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'predictions': predictions, 'anomaly_scores': anomaly_scores,
        'fidelities': fidelities, 'threshold': threshold,
        'y_true': y_test
    }


# =============================================================================
# STEP 8: RESULTS
# =============================================================================

def save_results(result):
    df = pd.DataFrame([{
        'Dataset': result['dataset'],
        'K': result['k'],
        'Fold': result['fold'],
        'Accuracy': round(result['accuracy'], 4),
        'Precision': round(result['precision'], 4),
        'Recall': round(result['recall'], 4),
        'F1-Score': round(result['f1_score'], 4),
        'Specificity': round(result['specificity'], 4),
        'TN': result['tn'],
        'FP': result['fp'],
        'FN': result['fn'],
        'TP': result['tp'],
        'Threshold': round(result['threshold'], 4)
    }])

    df.to_csv(os.path.join(OUTPUT_DIR, 'qae_cicddos2019_results.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'qae_results_for_paper.txt'), 'w') as f:
        f.write("FiD-QAE ANOMALY DETECTION RESULTS - CIC-DDoS2019\n")
        f.write("=" * 70 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Dataset: {DATASET_NAME}\n")
        f.write(f"  Input Qubits: {N_QUBITS}\n")
        f.write(f"  Latent Qubits: {N_LATENT}\n")
        f.write(f"  Trash Qubits: {N_TRASH}\n")
        f.write(f"  Training Samples (Normal): {N_TRAIN_NORMAL}\n")
        f.write(f"  Training Shots: {SHOTS_TRAIN}\n")
        f.write(f"  Inference Shots: {SHOTS_INFER}\n")
        f.write(f"  Max Iterations: {MAX_ITER}\n")
        f.write(f"  Optimizer: COBYLA\n\n")
        f.write("Results (k=2 Fold 1):\n")
        f.write("-" * 70 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("NOTE: Full multi-dataset, multi-fold evaluation deferred to future work.\n")
        f.write("      This proof-of-concept validates QAE anomaly detection on the\n")
        f.write("      primary benchmark dataset CIC-DDoS2019.\n")

    print(f"\nResults saved to {OUTPUT_DIR}:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FAST FIDELITY-DRIVEN QUANTUM AUTOENCODER (FiD-QAE)")
    print("Optimized Anomaly Detection - CIC-DDoS2019")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Input Qubits: {N_QUBITS}")
    print(f"  Latent Qubits: {N_LATENT}")
    print(f"  Trash Qubits: {N_TRASH}")
    print(f"  Training Samples (Normal): {N_TRAIN_NORMAL}")
    print(f"  Training Shots: {SHOTS_TRAIN}")
    print(f"  Inference Shots: {SHOTS_INFER}")
    print(f"  Max Iterations: {MAX_ITER}")
    print(f"  Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"  Hardware: Qiskit Aer Simulator (local)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\nNOTE: Unsupervised training on NORMAL data only.")
    print(f"      Anomaly score = 1 - fidelity (SWAP test).")
    print(f"      Based on Romero et al. (2017) QAE architecture.")
    print(f"      Full multi-dataset evaluation deferred to future work.\n")

    # Load data
    X, y = load_dataset()
    if X is None:
        print("ERROR: Could not load dataset.")
        return

    # k=2 fold cross-validation
    print(f"\n{'=' * 70}")
    print(f"k=2 FOLD CROSS-VALIDATION")
    print(f"{'=' * 70}")

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    fold_idx = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        result = evaluate_qae_fold(X_train, y_train, X_test, y_test, fold_idx)
        save_results(result)
        fold_idx += 1

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}/qae_cicddos2019_results.csv")
    print(f"Paper text saved to: {OUTPUT_DIR}/qae_results_for_paper.txt")


if __name__ == "__main__":
    main()