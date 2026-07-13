#!/usr/bin/env python3
"""
Unified Preprocessing for All 5 Standard Datasets
Maps all datasets to 6 consistent features: f1-f6
Ver: 3.0 | Date: 2026-07-06
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
import traceback

# Redirect all output to log file AND console
LOG_FILE = "/root/ddos-framework/preprocess_log.txt"
log_f = open(LOG_FILE, "w", buffering=1)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

OUTPUT_DIR = "/root/ddos-framework/data/standard_datasets/processed"
SAMPLES_PER_CLASS = 500
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def process_nslkdd():
    path = os.path.join(OUTPUT_DIR, "nsl-kdd_processed.csv")
    if not os.path.exists(path):
        log("ERROR: nsl-kdd_processed.csv not found!")
        return None
    df = pd.read_csv(path, header=None)
    log(f"NSL-KDD: {df.shape}, Labels: {df.iloc[:, -1].value_counts().to_dict()}")
    return df


def process_unsw_nb15():
    path = os.path.join(OUTPUT_DIR, "unsw-nb15_processed.csv")
    if not os.path.exists(path):
        log("ERROR: unsw-nb15_processed.csv not found!")
        return None
    df = pd.read_csv(path, header=None)
    log(f"UNSW-NB15: {df.shape}, Labels: {df.iloc[:, -1].value_counts().to_dict()}")
    return df


def process_cicddos2019():
    path = os.path.join(OUTPUT_DIR, "cicddos2019_processed.csv")
    if not os.path.exists(path):
        log("ERROR: cicddos2019_processed.csv not found!")
        return None
    df = pd.read_csv(path, header=None)
    log(f"CIC-DDoS2019: {df.shape}, Labels: {df.iloc[:, -1].value_counts().to_dict()}")
    return df


def process_cicids2017():
    """
    Process CICIDS2017 from Kaggle dataset.
    Process files ONE AT A TIME to save memory.
    """
    data_dir = "/root/ddos-framework/data/standard_datasets/cicids2017"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    csv_files = [f for f in csv_files if "pcap_ISCX" in f]

    log(f"Found {len(csv_files)} CICIDS2017 CSV files")

    # Process each file individually and extract features immediately
    # to avoid loading all into memory at once
    all_features = []
    total_rows = 0

    for csv_file in csv_files:
        log(f"\n  Reading {os.path.basename(csv_file)}...")
        try:
            # Read file
            df = pd.read_csv(csv_file)
            log(f"    Raw shape: {df.shape}")

            # Strip column names
            df.columns = df.columns.str.strip()

            # Check required columns exist
            required_cols = ['Label', 'Total Fwd Packets', 'Total Backward Packets',
                             'SYN Flag Count', 'ACK Flag Count', 'RST Flag Count',
                             'Flow IAT Mean', 'Flow Packets/s', 'Packet Length Mean',
                             'Packet Length Std']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                log(f"    ERROR: Missing columns: {missing}")
                continue

            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            log(f"    After cleaning: {len(df)}")

            # Label: BENIGN = 0, else = 1
            df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
            label_counts = df['Label'].value_counts().to_dict()
            log(f"    Labels: {label_counts}")

            # Extract f1-f6
            df['f1'] = df['Total Fwd Packets'] + df['Total Backward Packets']

            flag_sum = df['SYN Flag Count'] + df['ACK Flag Count'] + df['RST Flag Count']
            df['f2'] = np.where(flag_sum > 0, df['SYN Flag Count'] / flag_sum, 0)

            df['f3'] = df['Flow IAT Mean'].fillna(0)

            df['f4'] = df['Flow Packets/s'].replace([np.inf, -np.inf], 0).fillna(0)

            pkt_len_mean = df['Packet Length Mean']
            pkt_len_std = df['Packet Length Std']
            df['f5'] = np.where(pkt_len_mean > 0, pkt_len_std / (pkt_len_mean + 1), 0)
            df['f5'] = df['f5'].replace([np.inf, -np.inf], 0).fillna(0)

            df['f6'] = 0  # Placeholder

            # Select features
            features = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'Label']].copy()
            features = features.replace([np.inf, -np.inf], np.nan).dropna()

            log(f"    Features extracted: {features.shape}")
            all_features.append(features)
            total_rows += len(features)

            # Delete df to free memory
            del df

        except Exception as e:
            log(f"    ERROR processing {os.path.basename(csv_file)}: {str(e)}")
            traceback.print_exc(file=sys.stdout)

    log(f"\n  Combining {len(all_features)} feature sets...")
    if not all_features:
        log("  ERROR: No features extracted!")
        return None

    combined = pd.concat(all_features, ignore_index=True)
    log(f"  Combined shape: {combined.shape}")
    log(f"  Total rows: {total_rows}")

    # Balance dataset
    normal = combined[combined['Label'] == 0]
    attack = combined[combined['Label'] == 1]

    log(f"  Normal: {len(normal)}, Attack: {len(attack)}")

    if len(normal) == 0 or len(attack) == 0:
        log("  ERROR: Missing one class!")
        return None

    n_samples = min(SAMPLES_PER_CLASS, len(normal), len(attack))
    log(f"  Sampling {n_samples} from each class...")

    normal_sample = normal.sample(n=n_samples, random_state=RANDOM_STATE)
    attack_sample = attack.sample(n=n_samples, random_state=RANDOM_STATE)

    balanced = pd.concat([normal_sample, attack_sample], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    log(f"  Balanced: {balanced.shape}")
    log(f"  Labels: {balanced['Label'].value_counts().to_dict()}")

    output_path = os.path.join(OUTPUT_DIR, "cicids2017_processed.csv")
    balanced.to_csv(output_path, index=False, header=False)
    log(f"  Saved to {output_path}")

    return balanced


def process_kddcup1999():
    data_dir = "/root/ddos-framework/data/standard_datasets/kddcup1999"

    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]

    data_file = os.path.join(data_dir, "kddcup.data_10_percent", "kddcup.data_10_percent")
    log(f"Reading {data_file}...")

    df = pd.read_csv(data_file, names=column_names, header=None)
    log(f"  Total records: {len(df)}")

    df['label'] = df['label'].apply(lambda x: 0 if str(x).strip() == 'normal.' else 1)
    log(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    df['f1'] = df['count']
    df['f2'] = df['serror_rate']
    df['f3'] = df['dst_host_count']
    df['f4'] = df['srv_count']
    df['f5'] = 1.0 - df['dst_host_same_src_port_rate']
    df['f6'] = (df['protocol_type'] == 'udp').astype(int)

    features_df = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'label']].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()

    normal = features_df[features_df['label'] == 0]
    attack = features_df[features_df['label'] == 1]

    log(f"  Normal: {len(normal)}, Attack: {len(attack)}")

    n_samples = min(SAMPLES_PER_CLASS, len(normal), len(attack))
    normal_sample = normal.sample(n=n_samples, random_state=RANDOM_STATE)
    attack_sample = attack.sample(n=n_samples, random_state=RANDOM_STATE)

    balanced = pd.concat([normal_sample, attack_sample], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    log(f"  Balanced: {balanced.shape}")
    log(f"  Labels: {balanced['label'].value_counts().to_dict()}")

    output_path = os.path.join(OUTPUT_DIR, "kddcup1999_processed.csv")
    balanced.to_csv(output_path, index=False, header=False)
    log(f"  Saved to {output_path}")

    return balanced


if __name__ == "__main__":
    log("=" * 70)
    log("UNIFIED PREPROCESSING: ALL 5 DATASETS")
    log("=" * 70)

    datasets = {}

    log("\n" + "-" * 70)
    log("DATASET 1: NSL-KDD")
    log("-" * 70)
    datasets['nsl-kdd'] = process_nslkdd()

    log("\n" + "-" * 70)
    log("DATASET 2: UNSW-NB15")
    log("-" * 70)
    datasets['unsw-nb15'] = process_unsw_nb15()

    log("\n" + "-" * 70)
    log("DATASET 3: CIC-DDoS2019")
    log("-" * 70)
    datasets['cicddos2019'] = process_cicddos2019()

    log("\n" + "-" * 70)
    log("DATASET 4: CICIDS2017")
    log("-" * 70)
    datasets['cicids2017'] = process_cicids2017()

    log("\n" + "-" * 70)
    log("DATASET 5: KDD Cup 1999")
    log("-" * 70)
    datasets['kddcup1999'] = process_kddcup1999()

    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    for name, df in datasets.items():
        if df is not None:
            log(f"{name:15s}: {df.shape}, Labels: {df.iloc[:, -1].value_counts().to_dict()}")
        else:
            log(f"{name:15s}: ERROR - NOT FOUND")

    log("\n" + "=" * 70)
    log("ALL DONE!")
    log("=" * 70)

    # Pause so user can see output
    input("\nPress Enter to exit...")