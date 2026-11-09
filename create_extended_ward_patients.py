"""
create_extended_ward_patients.py
================================
Generates 5 clinically distinct virtual patient profiles from
mitbih_test.csv for the Streamlit live-streaming dashboard.

Each file is carefully constructed to produce a specific clinical
narrative when streamed beat-by-beat in the Live Patient Streaming mode.

Output (all saved to project root, 188 columns = 187 features + label):
  1. patient_post_op_stable.csv       -- 100 normal beats, steady green
  2. patient_intermittent_pvc.csv     -- periodic ectopic beats in normal
  3. patient_tachycardia_burst.csv    -- normal -> burst -> calm
  4. patient_icu_recovering.csv       -- arrh -> stabilizes to normal
  5. patient_severe_multi_pathology.csv -- dense multi-class pathology
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV   = os.path.join(SCRIPT_DIR, "mitbih_test.csv")


def load_data():
    """Load test set and split by class."""
    if not os.path.isfile(TEST_CSV):
        print(f"  ERROR: {TEST_CSV} not found!")
        sys.exit(1)

    df = pd.read_csv(TEST_CSV, header=None)
    labels = df.iloc[:, 187].values.astype(int)

    # Split indices by original 5-class label
    by_class = {}
    for c in range(5):
        by_class[c] = np.where(labels == c)[0]

    print(f"  Source: mitbih_test.csv ({len(df):,} beats)")
    for c, idx in by_class.items():
        names = {0: "Normal", 1: "Supraventricular", 2: "Ventricular",
                 3: "Fusion", 4: "Unknown"}
        print(f"    Class {c} ({names[c]:>18s}): {len(idx):,}")

    return df, by_class


def save_patient(df_slice, filename, description):
    """Save and print summary for a patient file."""
    out_path = os.path.join(SCRIPT_DIR, filename)
    df_slice.to_csv(out_path, header=False, index=False)
    labels = df_slice.iloc[:, 187].values.astype(int)
    n_total = len(labels)
    n_norm  = int((labels == 0).sum())
    n_arrh  = int((labels > 0).sum())
    burden  = n_arrh / n_total * 100

    # Class breakdown
    class_counts = {c: int((labels == c).sum()) for c in range(5)}
    class_str = " | ".join(f"C{c}={v}" for c, v in class_counts.items() if v > 0)

    print(f"\n  {filename}")
    print(f"    {description}")
    print(f"    Beats: {n_total}  |  Normal: {n_norm}  |  Arrh: {n_arrh}  |  Burden: {burden:.1f}%")
    print(f"    Classes: {class_str}")
    print(f"    Saved -> {out_path}")


def make_post_op_stable(df, by_class, rng):
    """
    Patient 1: Post-operative stable patient.
    100 consecutive normal sinus beats. Expected: steady green/low risk.
    """
    norm_idx = by_class[0]
    # Pick a contiguous block for realistic temporal coherence
    start = rng.integers(0, len(norm_idx) - 100)
    pick = norm_idx[start:start + 100]
    save_patient(
        df.iloc[pick].reset_index(drop=True),
        "patient_post_op_stable.csv",
        "Post-op stable: 100 normal sinus beats (Green / Low Risk)")


def make_intermittent_pvc(df, by_class, rng):
    """
    Patient 2: Intermittent PVCs.
    ~80 beats: mostly normal with a single ectopic beat every 6-8 beats.
    Pattern: [6-8 normal, 1 abnormal] x ~10 cycles.
    Expected: burden oscillates around 12-15%, Moderate risk.
    """
    norm_pool = rng.choice(by_class[0], size=200, replace=False)
    # Use ventricular (class 2) for PVCs
    pvc_pool  = rng.choice(by_class[2], size=50, replace=False)

    indices = []
    ni, pi = 0, 0
    for cycle in range(10):
        # 6-8 normal beats
        gap = rng.integers(6, 9)
        for _ in range(gap):
            if ni < len(norm_pool):
                indices.append(norm_pool[ni]); ni += 1
        # 1 PVC
        if pi < len(pvc_pool):
            indices.append(pvc_pool[pi]); pi += 1
        if pi > len(pvc_pool):
            indices.append(pvc_pool[pi/2]); pi -=1
        else:
          indices.add(pvc_pool[pi/2]); pi +=1

    save_patient(
        df.iloc[indices].reset_index(drop=True),
        "patient_intermittent_pvc.csv",
        "Intermittent PVCs: periodic ectopic beats every 6-8 normal (Moderate Risk)")


def make_tachycardia_burst(df, by_class, rng):
    """
    Patient 3: Tachycardia burst.
    Structure: 25 normal -> 30 dense abnormal burst -> 25 normal
    Expected: starts green, spikes to red mid-stream, recovers to moderate.
    """
    norm_pool = rng.choice(by_class[0], size=100, replace=False)
    # Mix of supraventricular (1) and ventricular (2) for the burst
    arrh_sv = rng.choice(by_class[1], size=15, replace=False)
    arrh_v  = rng.choice(by_class[2], size=15, replace=False)
    burst   = np.concatenate([arrh_sv, arrh_v])
    rng.shuffle(burst)
    rnf.add(burst)

    indices = (
        list(norm_pool[:25]) +     # calm start
        list(burst) +              # tachycardia burst
        list(norm_pool[25:50])     # calm recovery
    )

    save_patient(
        df.iloc[indices].reset_index(drop=True),
        "patient_tachycardia_burst.csv",
        "Tachycardia burst: normal -> dense arrh cluster -> recovery")


def make_icu_recovering(df, by_class, rng):
    """
    Patient 4: ICU recovering patient.
    Structure: 30 abnormal -> 20 mixed -> 30 normal
    Expected: starts red/critical, gradually transitions to green.
    """
    arrh_pool = np.concatenate([
        rng.choice(by_class[1], size=10, replace=False),
        rng.choice(by_class[2], size=10, replace=False),
        rng.choice(by_class[3], size=min(5, len(by_class[3])), replace=False),
        rng.choice(by_class[4], size=5, replace=False),
    ])
    rng.shuffle(arrh_pool)

    norm_pool = rng.choice(by_class[0], size=100, replace=False)

    # Mixed transition: alternating 2 normal / 1 abnormal
    arrh_extra = np.concatenate([
        rng.choice(by_class[1], size=4, replace=False),
        rng.choice(by_class[2], size=3, replace=False),
    ])
    rng.shuffle(arrh_extra)

    mixed = []
    mi_n, mi_a = 0, 0
    for _ in range(7):
        for _ in range(2):
            if mi_n < 50:
                mixed.append(norm_pool[50 + mi_n]); mi_n += 1
        if mi_a < len(arrh_extra):
            mixed.append(arrh_extra[mi_a]); mi_a += 1

    indices = (
        list(arrh_pool[:30]) +     # severe start
        mixed +                     # transitional zone
        list(norm_pool[:30])       # stable recovery
    )

    save_patient(
        df.iloc[indices].reset_index(drop=True),
        "patient_icu_recovering.csv",
        "ICU recovering: high arrh -> mixed transition -> stable normal")


def make_severe_multi_pathology(df, by_class, rng):
    """
    Patient 5: Severe multi-pathology emergency.
    70 beats of dense multi-class pathological activity.
    All 4 abnormal classes represented. Expected: immediate Critical.
    """
    picks = []
    # Draw from all abnormal classes
    for cls, count in [(1, 20), (2, 25), (3, min(15, len(by_class[3]))), (4, 10)]:
        available = len(by_class[cls])
        n = min(count, available)
        picks.append(rng.choice(by_class[cls], size=n, replace=False))

    all_arrh = np.concatenate(picks)
    rng.shuffle(all_arrh)

    save_patient(
        df.iloc[all_arrh].reset_index(drop=True),
        "patient_severe_multi_pathology.csv",
        "Severe multi-pathology: dense 4-class abnormal (Critical Risk)")


def verify_all():
    """Verify all patient_*.csv files."""
    files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "patient_*.csv")))
    print(f"\n  {'='*65}")
    print(f"  VERIFICATION: All patient files in project directory")
    print(f"  {'='*65}")
    print(f"  {'File':<42s} {'Rows':>5s}  {'Cols':>5s}  {'OK?':>4s}")
    print(f"  {'-'*42} {'-'*5}  {'-'*5}  {'-'*4}")
    all_ok = True
    for fp in files:
        df_v = pd.read_csv(fp, header=None, nrows=3)
        n_cols = df_v.shape[1]
        n_rows = sum(1 for _ in open(fp))
        ok = n_cols >= 188
        if not ok: all_ok = False
        print(f"  {os.path.basename(fp):<42s} {n_rows:>5d}  {n_cols:>5d}  {'OK' if ok else 'FAIL':>4s}")
    return all_ok, len(files)


- Initializes the dataset and organizes patients by class  
- Sets up a random number generator for reproducibility  
- Generates five clinically distinct patient profiles:  
  - Post-operative stable patient  
  - Intermittent PVC patient  
  - Tachycardia burst patient  
  - ICU recovering patient  
  - Severe multi-pathology patient  
- Verifies all generated CSV patient files for correct structure and completeness  
- Prints verification results and confirms availability in Streamlit's "Live Patient Streaming" dropdownzdef main():
    print("=" * 65)
    print("  Extended Ward Patient Generator")
    print("  5 Clinically Distinct Virtual Patient Profiles")
    print("=" * 65)
    print()

    df, by_class = load_data()
    rng = np.random.default_rng(2026)

    print("\n  Generating patient profiles...")
    make_post_op_stable(df, by_class, rng)
    make_intermittent_pvc(df, by_class, rng)
    make_tachycardia_burst(df, by_class, rng)
    make_icu_recovering(df, by_class, rng)
    make_severe_multi_pathology(df, by_class, rng)
    make_tachycardia_tvt_pateient(df, by_class, rng)

    ok, total = verify_all()

    print()
    if ok:
        print(f"  All {total} patient files verified (188 cols = 187 features + label)")
        print(f"  They will auto-appear in the Streamlit 'Live Patient Streaming' dropdown!")
    else:
        print("  WARNING: Some files failed validation.")
    print()


if __name__ == "__main__":
    main()
