"""
create_patients.py
==================
Slices mitbih_test.csv into two virtual patient datasets for live demo:

  patient_low_risk.csv  -- 60 consecutive normal beats (healthy profile)
  patient_high_risk.csv -- ~60 beats, majority arrhythmic + a few normal
                           (high-risk cardiac session)

Output format: 188 columns (187 voltage features + label), no header,
matching the MIT-BIH CSV structure expected by the Streamlit uploader.
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV   = os.path.join(SCRIPT_DIR, "mitbih_test.csv")

def main():
    print("Loading mitbih_test.csv ...")
    df = pd.read_csv(TEST_CSV, header=None)
    X      = df.iloc[:, :187]
    labels = df.iloc[:, 187].values.astype(int)

    normal_idx = np.where(labels == 0)[0]
    arrh_idx   = np.where(labels > 0)[0]

    print(f"  Total beats : {len(labels):,}")
    print(f"  Normal      : {len(normal_idx):,}")
    print(f"  Arrhythmic  : {len(arrh_idx):,}")

    # ------------------------------------------------------------------
    # Patient 1 — Low Risk: 60 consecutive normal beats
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    start = rng.integers(0, len(normal_idx) - 60)
    low_risk_idx = normal_idx[start : start + 60]
    df_low = df.iloc[low_risk_idx].reset_index(drop=True)

    out_low = os.path.join(SCRIPT_DIR, "patient_low_risk.csv")
    df_low.to_csv(out_low, header=False, index=False)
    n_arrh_low = int((df_low.iloc[:, 187].values > 0).sum())
    print(f"\n  patient_low_risk.csv")
    print(f"    Beats     : {len(df_low)}")
    print(f"    Normal    : {len(df_low) - n_arrh_low}")
    print(f"    Arrhythmic: {n_arrh_low}")
    print(f"    Saved  -> {out_low}")

    # ------------------------------------------------------------------
    # Patient 2 — High Risk: ~45 arrhythmic + ~15 normal (mixed)
    # ------------------------------------------------------------------
    n_arrh_pick  = 45
    n_norm_pick  = 15
    picked_arrh  = rng.choice(arrh_idx, size=n_arrh_pick, replace=False)
    picked_norm  = rng.choice(normal_idx, size=n_norm_pick, replace=False)
    high_risk_idx = np.concatenate([picked_arrh, picked_norm])
    rng.shuffle(high_risk_idx)            # interleave for realism
    df_high = df.iloc[high_risk_idx].reset_index(drop=True)

    out_high = os.path.join(SCRIPT_DIR, "patient_high_risk.csv")
    df_high.to_csv(out_high, header=False, index=False)
    n_arrh_high = int((df_high.iloc[:, 187].values > 0).sum())
    print(f"\n  patient_high_risk.csv")
    print(f"    Beats     : {len(df_high)}")
    print(f"    Normal    : {len(df_high) - n_arrh_high}")
    print(f"    Arrhythmic: {n_arrh_high}")
    print(f"    Saved  -> {out_high}")

    print("\nDone! Both files are ready for the Streamlit uploader.")


if __name__ == "__main__":
    main()
