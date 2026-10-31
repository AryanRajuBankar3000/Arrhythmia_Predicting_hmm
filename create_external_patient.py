"""
create_external_patient.py
==========================
Processes the PTB Diagnostic Database (ptbdb_abnormal.csv) to generate
a demo patient file for the Streamlit dashboard.

PTB DB format: 187 voltage columns + 1 label column (same structure as
MIT-BIH). Abnormal records represent various cardiac pathologies
(myocardial infarction, cardiomyopathy, bundle branch block, etc.).

Output:
    patient_external_ptb.csv  --  50 pathological heartbeats, ready for
                                  the Streamlit file uploader.
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PTB_CSV    = os.path.join(SCRIPT_DIR, "ptbdb_abnormal.csv")
OUT_CSV    = os.path.join(SCRIPT_DIR, "patient_external_ptb.csv")
N_BEATS    = 50


def main():
    # ── Step 1: Check if source file exists ──────────────────────────────
    print("=" * 65)
    print("  PTB Diagnostic Database -> External Patient Generator")
    print("=" * 65)

    if not os.path.isfile(PTB_CSV):
        print(f"\n  ERROR: '{os.path.basename(PTB_CSV)}' not found in:")
        print(f"         {SCRIPT_DIR}")
        print()
        print("  Please download it from Kaggle and place it here:")
        print("  https://www.kaggle.com/datasets/shayanfazeli/heartbeat")
        print("  (file: ptbdb_abnormal.csv)")
        sys.exit(1)

    # ── Step 2: Load the dataset ─────────────────────────────────────────
    print(f"\n  Loading {os.path.basename(PTB_CSV)} ...")
    df = pd.read_csv(PTB_CSV, header=None)
    n_rows, n_cols = df.shape
    print(f"  Rows   : {n_rows:,}")
    print(f"  Columns: {n_cols}")

    # Validate structure
    if n_cols < 187:
        print(f"\n  ERROR: Expected at least 187 columns, got {n_cols}.")
        sys.exit(1)

    # Identify voltage features and optional label
    X = df.iloc[:, :187]
    has_label = n_cols >= 188
    if has_label:
        labels = df.iloc[:, 187].values.astype(int)
        n_abnormal = int((labels > 0).sum())
        n_normal   = int((labels == 0).sum())
        print(f"  Label column detected:")
        print(f"    Normal   : {n_normal:,}")
        print(f"    Abnormal : {n_abnormal:,}")
    else:
        print("  No label column detected — treating all rows as abnormal.")

    # ── Step 3: Extract a representative slice of 50 beats ───────────────
    rng = np.random.default_rng(42)

    if has_label:
        # Prefer abnormal rows; fall back to all rows if fewer than N_BEATS
        abnormal_idx = np.where(labels > 0)[0]
        if len(abnormal_idx) >= N_BEATS:
            pick = rng.choice(abnormal_idx, size=N_BEATS, replace=False)
        else:
            pick = rng.choice(len(df), size=min(N_BEATS, len(df)), replace=False)
    else:
        pick = rng.choice(len(df), size=min(N_BEATS, len(df)), replace=False)

    pick.sort()  # keep temporal order
    df_slice = df.iloc[pick].reset_index(drop=True)

    # ── Step 4: Save output CSV ──────────────────────────────────────────
    df_slice.to_csv(OUT_CSV, header=False, index=False)

    # Summary
    if has_label:
        slice_labels = df_slice.iloc[:, 187].values.astype(int)
        s_norm = int((slice_labels == 0).sum())
        s_abn  = int((slice_labels > 0).sum())
    else:
        s_norm, s_abn = 0, len(df_slice)

    print(f"\n  patient_external_ptb.csv")
    print(f"    Beats     : {len(df_slice)}")
    print(f"    Normal    : {s_norm}")
    print(f"    Abnormal  : {s_abn}")
    print(f"    Columns   : {df_slice.shape[1]} (187 voltage + label)")
    print(f"    Saved  -> {OUT_CSV}")

    # ── Step 5: Confirmation ─────────────────────────────────────────────
    print()
    print("  " + "-" * 55)
    print("  File is ready for the Streamlit dashboard uploader!")
    print("  " + "-" * 55)
    print()


if __name__ == "__main__":
    main()
