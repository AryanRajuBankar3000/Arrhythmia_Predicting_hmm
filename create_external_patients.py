"""
create_external_patients.py
============================
Generates two distinct virtual patient CSV files for the Streamlit
live-streaming dashboard.

Primary source: ptbdb_abnormal.csv (PTB Diagnostic Database)
Fallback:       mitbih_test.csv   (MIT-BIH Arrhythmia Database)

Output files (saved to project directory):
  patient_external_ptb_acute.csv  -- 80 severe abnormal beats (high-risk)
  patient_external_ptb_mixed.csv  -- 60 beats alternating normal/abnormal
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PTB_CSV    = os.path.join(SCRIPT_DIR, "ptbdb_abnormal.csv")
MITBIH_CSV = os.path.join(SCRIPT_DIR, "mitbih_test.csv")

OUT_ACUTE  = os.path.join(SCRIPT_DIR, "patient_external_ptb_acute.csv")
OUT_MIXED  = os.path.join(SCRIPT_DIR, "patient_external_ptb_mixed.csv")


def load_source():
    """Load ptbdb_abnormal.csv if available, else fall back to mitbih_test.csv."""
    if os.path.isfile(PTB_CSV):
        print(f"  Source: ptbdb_abnormal.csv (PTB Diagnostic Database)")
        df = pd.read_csv(PTB_CSV, header=None)
        source = "PTB"
    elif os.path.isfile(MITBIH_CSV):
        print(f"  Source: mitbih_test.csv (MIT-BIH fallback)")
        print(f"  NOTE:  ptbdb_abnormal.csv not found -- using MIT-BIH test set instead.")
        print(f"         Place ptbdb_abnormal.csv in the project folder and re-run")
        print(f"         to generate true cross-dataset patient files.\n")
        df = pd.read_csv(MITBIH_CSV, header=None)
        source = "MIT-BIH"
    else:
        print("  ERROR: Neither ptbdb_abnormal.csv nor mitbih_test.csv found!")
        sys.exit(1)

    n_rows, n_cols = df.shape
    print(f"  Rows   : {n_rows:,}")
    print(f"  Columns: {n_cols}")
    return df, source


def make_acute_patient(df, rng):
    """
    Patient A -- Acute cardiac event.
    80 beats of the most extreme arrhythmic morphology (highest voltage peaks).
    Simulates a patient in acute arrhythmic distress.
    """
    labels = df.iloc[:, 187].values.astype(int)
    abnormal_idx = np.where(labels > 0)[0]

    if len(abnormal_idx) < 80:
        # If not enough abnormal, just take what we have
        pick = abnormal_idx
    else:
        # Pick the 80 abnormal beats with the highest peak voltage (most
        # morphologically extreme = most visually dramatic for the demo)
        peak_voltages = df.iloc[abnormal_idx, :187].max(axis=1).values
        top_80 = np.argsort(peak_voltages)[-80:]
        pick = abnormal_idx[top_80]

    rng.shuffle(pick)
    df_acute = df.iloc[pick].reset_index(drop=True)
    df_acute.to_csv(OUT_ACUTE, header=False, index=False)

    sl = df_acute.iloc[:, 187].values.astype(int)
    n_norm = int((sl == 0).sum())
    n_abn  = int((sl > 0).sum())
    print(f"\n  patient_external_ptb_acute.csv")
    print(f"    Beats     : {len(df_acute)}")
    print(f"    Normal    : {n_norm}")
    print(f"    Abnormal  : {n_abn}")
    print(f"    Burden    : {n_abn / len(df_acute) * 100:.1f}%")
    print(f"    Saved  -> {OUT_ACUTE}")
    return df_acute


def make_mixed_patient(df, rng):
    """
    Patient B -- Intermittent arrhythmia (mixed).
    60 beats with alternating blocks of normal and abnormal beats.
    Simulates a patient with paroxysmal arrhythmia -- the risk banner
    will visibly oscillate during streaming.
    """
    labels = df.iloc[:, 187].values.astype(int)
    normal_idx   = np.where(labels == 0)[0]
    abnormal_idx = np.where(labels > 0)[0]

    # Build alternating blocks: 8 normal, 4 abnormal, repeated
    mixed_indices = []
    n_target = 60
    norm_ptr = rng.choice(len(normal_idx), size=min(200, len(normal_idx)), replace=False)
    abn_ptr  = rng.choice(len(abnormal_idx), size=min(200, len(abnormal_idx)), replace=False)
    ni, ai = 0, 0

    while len(mixed_indices) < n_target:
        # Normal block (8 beats)
        block_n = min(8, n_target - len(mixed_indices), len(norm_ptr) - ni)
        if block_n > 0:
            mixed_indices.extend(normal_idx[norm_ptr[ni:ni + block_n]])
            ni += block_n

        # Abnormal block (4 beats)
        block_a = min(4, n_target - len(mixed_indices), len(abn_ptr) - ai)
        if block_a > 0:
            mixed_indices.extend(abnormal_idx[abn_ptr[ai:ai + block_a]])
            ai += block_a

        if ni >= len(norm_ptr) and ai >= len(abn_ptr):
            break

    mixed_indices = np.array(mixed_indices[:n_target])
    df_mixed = df.iloc[mixed_indices].reset_index(drop=True)
    df_mixed.to_csv(OUT_MIXED, header=False, index=False)

    sl = df_mixed.iloc[:, 187].values.astype(int)
    n_norm = int((sl == 0).sum())
    n_abn  = int((sl > 0).sum())
    print(f"\n  patient_external_ptb_mixed.csv")
    print(f"    Beats     : {len(df_mixed)}")
    print(f"    Normal    : {n_norm}")
    print(f"    Abnormal  : {n_abn}")
    print(f"    Burden    : {n_abn / len(df_mixed) * 100:.1f}%")
    print(f"    Saved  -> {OUT_MIXED}")
    return df_mixed


def verify_files():
    """Verify all patient_*.csv files have exactly 188 columns (187 features + 1 label)."""
    import glob
    patient_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "patient_*.csv")))
    print(f"\n  Verification -- all patient files in project directory:")
    print(f"  {'File':<40s} {'Rows':>6s}  {'Cols':>5s}  {'Status':>8s}")
    print(f"  {'-'*40} {'-'*6}  {'-'*5}  {'-'*8}")
    all_ok = True
    for fp in patient_files:
        df_v = pd.read_csv(fp, header=None, nrows=5)
        n_cols = df_v.shape[1]
        n_rows = sum(1 for _ in open(fp))
        ok = n_cols >= 188
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {os.path.basename(fp):<40s} {n_rows:>6d}  {n_cols:>5d}  {status:>8s}")
    return all_ok


def main():
    print("=" * 65)
    print("  External Patient File Generator")
    print("=" * 65)

    df, source = load_source()
    rng = np.random.default_rng(123)

    make_acute_patient(df, rng)
    make_mixed_patient(df, rng)

    print()
    ok = verify_files()

    print()
    if ok:
        print("  " + "-" * 55)
        print("  All files ready! They will appear in the Streamlit")
        print("  sidebar dropdown under 'Live Patient Streaming'.")
        print("  " + "-" * 55)
    else:
        print("  WARNING: Some files failed column validation.")
    print()


if __name__ == "__main__":
    main()
