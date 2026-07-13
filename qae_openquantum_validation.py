#!/usr/bin/env python3
"""
================================================================================
QAE OPEN QUANTUM REAL HARDWARE VALIDATION
Submits a representative QAE circuit to IonQ Forte-1 trapped-ion QPU
For: Quantum-based Multi-Vector DDoS Attack Detection Framework

HARDWARE: IonQ Forte-1 via Open Quantum Platform
CREDITS: Uses ~5-10 Spark credits (Public execution plan)
NOTE: This is a small validation only. Full evaluation is on simulator.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Open Quantum SDK
from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import SchedulerClient, JobSubmissionConfig

# Qiskit for circuit building
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.qasm2 import dumps

# =============================================================================
# CONFIGURATION
# =============================================================================

N_FEATURES = 6
N_STATE_QUBITS = 3
N_EVAL_QUBITS = 3
SHOTS = 1024  # Reduced for cost efficiency
THRESHOLD = 0.5

# Load a sample from CIC-DDoS2019 for validation
DATA_DIR = "/root/ddos-framework/data/standard_datasets/processed"
SAMPLE_FILE = "cicddos2019_processed.csv"


# =============================================================================
# STEP 1: LOAD AND NORMALIZE SAMPLE
# =============================================================================

def load_sample():
    filepath = os.path.join(DATA_DIR, SAMPLE_FILE)
    df = pd.read_csv(filepath, header=None)

    # Take first attack sample (label = 1)
    attack_samples = df[df.iloc[:, 6] == 1].iloc[:5, :6].values
    normal_samples = df[df.iloc[:, 6] == 0].iloc[:5, :6].values

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    all_samples = np.vstack([attack_samples, normal_samples])
    scaled = scaler.fit_transform(all_samples)

    labels = [1] * 5 + [0] * 5
    return scaled, labels


# =============================================================================
# STEP 2: BUILD QAE CIRCUIT
# =============================================================================

def amplitude_encode(features):
    padded = np.zeros(2 ** N_STATE_QUBITS)
    padded[:N_FEATURES] = features
    norm = np.linalg.norm(padded)
    if norm > 1e-10:
        padded = padded / norm
    else:
        padded[0] = 1.0
    return padded


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


def build_qae_circuit_qasm(features):
    encoded = amplitude_encode(features)
    n_state = N_STATE_QUBITS
    n_eval = N_EVAL_QUBITS

    eval_reg = QuantumRegister(n_eval, 'eval')
    state_reg = QuantumRegister(n_state, 'state')
    c_reg = ClassicalRegister(n_eval, 'c')
    qc = QuantumCircuit(eval_reg, state_reg, c_reg)

    init_circ = QuantumCircuit(n_state)
    init_circ.initialize(encoded, range(n_state))
    init_circ = init_circ.decompose(reps=2)
    qc.compose(init_circ, qubits=state_reg, inplace=True)

    qc.h(eval_reg)

    oracle = build_oracle(n_state)
    diffuser = build_diffuser(n_state)

    for j in range(n_eval):
        repetitions = 2 ** j
        for _ in range(repetitions):
            qc.compose(oracle, qubits=state_reg, inplace=True)
            qc.compose(diffuser, qubits=state_reg, inplace=True)

    qft_inv = QFT(n_eval, inverse=True)
    qc.compose(qft_inv, qubits=eval_reg, inplace=True)
    qc.measure(eval_reg, c_reg)

    return dumps(qc)


# =============================================================================
# STEP 3: SUBMIT TO OPEN QUANTUM
# =============================================================================

def submit_to_openquantum(qasm_str, job_name):
    client_id = os.environ.get("OPENQUANTUM_CLIENT_ID")
    client_secret = os.environ.get("OPENQUANTUM_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("ERROR: Set OPENQUANTUM_CLIENT_ID and OPENQUANTUM_CLIENT_SECRET")
        print("  export OPENQUANTUM_CLIENT_ID='s_your_id'")
        print("  export OPENQUANTUM_CLIENT_SECRET='your_secret'")
        return None

    auth = ClientCredentialsAuth(
        creds=ClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        ),
    )

    scheduler = SchedulerClient(auth=auth)

    config = JobSubmissionConfig(
        backend_class_id="ionq:forte-1",
        name=job_name,
        job_subcategory_id="phys:oth",
        shots=SHOTS,
    )

    print(f"Submitting job: {job_name}")
    print(f"Backend: IonQ Forte-1")
    print(f"Shots: {SHOTS}")

    job = scheduler.submit_job(config, qasm_str=qasm_str)

    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Estimated cost: Check portal for credit consumption")

    scheduler.close()
    return job


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("QAE OPEN QUANTUM REAL HARDWARE VALIDATION")
    print("IonQ Forte-1 Trapped-Ion QPU")
    print("=" * 70)

    # Load samples
    samples, labels = load_sample()
    print(f"\nLoaded {len(samples)} samples from CIC-DDoS2019")
    print(f"Labels: {labels}")

    # Submit first attack sample
    print("\n--- Submitting Attack Sample (Sample 0) ---")
    qasm_attack = build_qae_circuit_qasm(samples[0])
    job_attack = submit_to_openquantum(qasm_attack, "QAE_DDoS_Attack_Validation")

    # Submit first normal sample
    print("\n--- Submitting Normal Sample (Sample 5) ---")
    qasm_normal = build_qae_circuit_qasm(samples[5])
    job_normal = submit_to_openquantum(qasm_normal, "QAE_DDoS_Normal_Validation")

    print("\n" + "=" * 70)
    print("JOBS SUBMITTED")
    print("=" * 70)
    print("\nCheck job status at: https://portal.openquantum.com/jobs")
    print("\nAfter jobs complete, download results and verify:")
    print("  - Attack sample should show high anomaly probability")
    print("  - Normal sample should show low anomaly probability")
    print("\nThis validates that QAE circuits execute correctly on real quantum hardware.")


if __name__ == "__main__":
    main()  