#!/usr/bin/env python3
"""
================================================================================
FIDELITY-DRIVEN QUANTUM AUTOENCODER (FiD-QAE) FOR ANOMALY DETECTION
Full Evaluation on 5 Standard Datasets - Qiskit Aer Simulator
For: Quantum-based Multi-Vector DDoS Attack Detection Framework

APPROACH: Unsupervised QAE trained on NORMAL data only.
Anomaly score = 1 - fidelity (low fidelity = anomaly).
Based on: Romero et al. (2017), Qiskit Machine Learning Tutorial (2024),
          FiD-QAE for Credit Card Fraud Detection (2024).

HARDWARE: Qiskit Aer Simulator (local)
NOTE: Real quantum hardware validation on Open Quantum IonQ Forte-1
      performed separately.
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

N_FEATURES = 6  # Number of classical features
N_QUBITS = 3  # ceil(log2(6)) = 3 qubits for amplitude encoding
N_LATENT = 2  # Compressed latent qubits
N_TRASH = 1  # Trash qubits (N_QUBITS - N_LATENT)
SHOTS = 8192
RANDOM_STATE = 42
K_FOLDS = [2, 3, 4, 5]
MAX_ITER = 50  # COBYLA optimizer iterations for training (reduced for speed)

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
    X = df.iloc[:, :N_FEATURES].values.astype(float)
    y = df.iloc[:, N_FEATURES].values.astype(int)

    print(f"  Loaded {name}: {X.shape[0]} samples, {X.shape[1]} features")
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
    """
    Encode 6 features into 3-qubit quantum state amplitudes.
    Pad with 2 zeros to make 8 amplitudes (2^3).
    Normalize to unit vector.
    """
    padded = np.zeros(2 ** N_QUBITS, dtype=complex)
    padded[:N_FEATURES] = features
    norm = np.linalg.norm(padded)
    if norm > 1e-10:
        padded = padded / norm
    else:
        padded[0] = 1.0
    return padded


# =============================================================================
# STEP 4: QAE CIRCUIT COMPONENTS
# =============================================================================

def build_encoder_ansatz(n_qubits, n_params):
    """
    Build a hardware-efficient parameterized ansatz using ParameterVector.
    Uses RY rotations and CNOT entanglement.
    """
    params = ParameterVector('θ', n_params)
    qc = QuantumCircuit(n_qubits, name='Encoder')

    param_idx = 0
    # Layer 1: RY rotations on all qubits
    for q in range(n_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1

    # Entanglement: CNOT chain
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)

    # Layer 2: RY rotations
    for q in range(n_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1

    # Entanglement: CNOT chain reverse
    for q in range(n_qubits - 1, 0, -1):
        qc.cx(q - 1, q)

    # Layer 3: RY rotations
    for q in range(n_qubits):
        qc.ry(params[param_idx], q)
        param_idx += 1

    return qc, params


def build_training_circuit(feature_vector, n_params):
    """
    Build QAE training circuit with SWAP test.

    Qubit layout (6 total):
    - q[0:3] : Input state (amplitude encoded, 3 qubits)
    - q[3]   : Trash state (after encoder, 1 qubit)
    - q[4]   : Reference state |0> (1 qubit)
    - q[5]   : Auxiliary qubit for SWAP test
    """
    total_qubits = N_QUBITS + N_TRASH + N_TRASH + 1  # 3 + 1 + 1 + 1 = 6
    aux_qubit = total_qubits - 1  # qubit 5
    ref_qubit = N_QUBITS + N_TRASH  # qubit 4 (initialized to |0>)
    trash_qubit = N_QUBITS  # qubit 3

    qr = QuantumRegister(total_qubits, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)

    # Step 1: Amplitude encode input features into qubits 0,1,2
    encoded = amplitude_encode(feature_vector)
    prep = StatePreparation(encoded)
    qc.compose(prep, qubits=list(range(N_QUBITS)), inplace=True)

    # Step 2: Apply parameterized encoder on all 3 input qubits
    encoder, params = build_encoder_ansatz(N_QUBITS, n_params)
    qc.compose(encoder, qubits=list(range(N_QUBITS)), inplace=True)

    # Step 3: SWAP test between trash qubit (3) and reference qubit (4)
    # Reference qubit is already |0>
    qc.h(aux_qubit)
    qc.cswap(aux_qubit, trash_qubit, ref_qubit)
    qc.h(aux_qubit)

    # Measure auxiliary qubit
    qc.measure(aux_qubit, cr[0])

    return qc, params


# =============================================================================
# STEP 5: TRAINING (Unsupervised - Normal Data Only)
# =============================================================================

def compute_fidelity(parameters, feature_vector, simulator, n_params):
    """
    Build circuit, bind parameters, run on simulator, compute fidelity.
    Fidelity = P(measuring |0> on auxiliary qubit)
    """
    qc, param_vector = build_training_circuit(feature_vector, n_params)

    # Create parameter dictionary
    param_dict = {param_vector[i]: parameters[i] for i in range(n_params)}

    # Bind parameters to circuit
    bound_qc = qc.assign_parameters(param_dict)

    # Transpile for AerSimulator
    transpiled_qc = transpile(bound_qc, simulator)

    # Run
    job = simulator.run(transpiled_qc, shots=SHOTS)
    result = job.result()
    counts = result.get_counts()

    # Fidelity = probability of measuring |0> on auxiliary qubit
    n_zero = counts.get('0', 0)
    fidelity = n_zero / SHOTS

    return fidelity


def train_qae(X_normal, simulator, max_iter=MAX_ITER):
    """
    Train QAE on NORMAL data only.
    Objective: Maximize fidelity between trash state and |0> reference.
    Loss = 1 - mean(fidelity)
    """
    n_samples = len(X_normal)
    n_params = N_QUBITS * 3  # 3 layers of RY rotations
    print(f"    Training QAE: {n_samples} normal samples, {n_params} parameters")

    def cost_function(params):
        total_fidelity = 0.0
        for x in X_normal:
            fidelity = compute_fidelity(params, x, simulator, n_params)
            total_fidelity += fidelity
        mean_fidelity = total_fidelity / n_samples
        loss = 1.0 - mean_fidelity
        return loss

    # Initialize parameters randomly
    initial_params = np.random.uniform(-np.pi, np.pi, n_params)

    print(f"    Starting optimization (max_iter={max_iter})...")
    start_time = time.time()

    # Use scipy minimize with COBYLA
    opt_result = minimize(
        cost_function,
        initial_params,
        method='COBYLA',
        options={'maxiter': max_iter, 'rhobeg': 0.1}
    )

    elapsed = time.time() - start_time
    print(f"    Training complete in {elapsed:.1f}s")
    print(f"    Final loss: {opt_result.fun:.6f}")
    print(f"    Final fidelity: {1.0 - opt_result.fun:.6f}")
    print(f"    Function evaluations: {opt_result.nfev}")

    return opt_result.x


# =============================================================================
# STEP 6: ANOMALY DETECTION (Inference)
# =============================================================================

def detect_anomaly(parameters, feature_vector, threshold, simulator, n_params):
    """
    Detect anomaly using trained QAE.
    Low fidelity = anomaly (does not compress well like normal data).
    """
    fidelity = compute_fidelity(parameters, feature_vector, simulator, n_params)
    anomaly_score = 1.0 - fidelity  # High score = anomaly
    prediction = 1 if anomaly_score > threshold else 0
    return prediction, anomaly_score, fidelity


def determine_threshold(parameters, X_normal_val, simulator, n_params):
    """
    Determine anomaly threshold from validation normal data.
    Use mean + 2*std of anomaly scores as threshold.
    """
    scores = []
    for x in X_normal_val:
        fidelity = compute_fidelity(parameters, x, simulator, n_params)
        scores.append(1.0 - fidelity)

    scores = np.array(scores)
    threshold = np.mean(scores) + 2 * np.std(scores)
    print(f"    Threshold determined: {threshold:.4f} (mean={np.mean(scores):.4f}, std={np.std(scores):.4f})")
    return threshold


# =============================================================================
# STEP 7: EVALUATION
# =============================================================================

def evaluate_qae_fold(X_train, y_train, X_test, y_test, fold_num, dataset_name):
    print(f"    Fold {fold_num}: train={len(X_train)}, test={len(X_test)}")

    # Split training data: use only NORMAL samples for training
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    print(f"    Normal training samples: {len(X_normal)}")

    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    X_normal_scaled = scaler.transform(X_normal)

    # Split normal data: 80% for training, 20% for threshold calibration
    split_idx = int(0.8 * len(X_normal_scaled))
    X_normal_train = X_normal_scaled[:split_idx]
    X_normal_val = X_normal_scaled[split_idx:]

    simulator = AerSimulator()
    n_params = N_QUBITS * 3

    # Step 1: Train QAE on normal data
    print(f"    Training QAE on normal data...")
    trained_params = train_qae(X_normal_train, simulator, max_iter=MAX_ITER)

    # Step 2: Determine threshold from validation normal data
    print(f"    Determining anomaly threshold...")
    threshold = determine_threshold(trained_params, X_normal_val, simulator, n_params)

    # Step 3: Detect anomalies on test set
    print(f"    Running anomaly detection on test set...")
    predictions = []
    anomaly_scores = []
    fidelities = []

    start_time = time.time()

    for i, x in enumerate(X_test_scaled):
        pred, score, fid = detect_anomaly(trained_params, x, threshold, simulator, n_params)
        predictions.append(pred)
        anomaly_scores.append(score)
        fidelities.append(fid)

        if (i + 1) % 25 == 0 or i == len(X_test_scaled) - 1:
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

    return {
        'fold': fold_num, 'k': None, 'dataset': dataset_name,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'specificity': specificity,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'predictions': predictions, 'anomaly_scores': anomaly_scores,
        'fidelities': fidelities, 'threshold': threshold,
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
# STEP 8: RESULTS AGGREGATION
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
            'Approach': 'FiD-QAE',
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
        f.write("FiD-QAE ANOMALY DETECTION RESULTS\n")
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
    print("FIDELITY-DRIVEN QUANTUM AUTOENCODER (FiD-QAE)")
    print("Anomaly Detection - Qiskit Aer Simulator")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input Qubits: {N_QUBITS}")
    print(f"  Latent Qubits: {N_LATENT}")
    print(f"  Trash Qubits: {N_TRASH}")
    print(f"  Total Circuit Qubits: {N_QUBITS + N_TRASH + N_TRASH + 1}")
    print(f"  Shots: {SHOTS}")
    print(f"  Optimizer: COBYLA (max_iter={MAX_ITER})")
    print(f"  Hardware: Qiskit Aer Simulator (local)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\nNOTE: Unsupervised training on NORMAL data only.")
    print(f"      Anomaly score = 1 - fidelity (SWAP test).")
    print(f"      Based on Romero et al. (2017) QAE architecture.\n")

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