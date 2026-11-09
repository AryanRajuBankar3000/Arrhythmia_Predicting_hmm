"""
================================================================================
 Cardiac Arrhythmia Diagnostic Pipeline -- Streamlit Web Application
 University Defense Demo
================================================================================
 Mode A -- Live Diagnostic (File Upload):
   Upload any MIT-BIH-format CSV (187 voltage columns + optional label col).
   The app runs PCA -> Full-Covariance HMM -> clinical risk report, all live.

 Mode B -- Full MIT-BIH Benchmark (legacy):
   Train on the bundled mitbih_train.csv and evaluate on mitbih_test.csv.
================================================================================
"""

import io, os, time as _time, glob
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report, confusion_matrix)

from arrhythmia_hmm import (
    run_pipeline, prepare_hmm_data, train_both, TRAIN_CSV,
)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardiac Arrhythmia Diagnostic -- HMM",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { font-family: "Inter", sans-serif; box-sizing: border-box; }
.stApp { background-color: #0d1117; color: #e6edf3; }
h1, h2, h3, h4 { color: #58a6ff !important; }
.stMarkdown p { color: #c9d1d9; }

.hero-box {
    background: linear-gradient(135deg, #161b22 0%, #1a2332 100%);
    padding: 28px 32px; border-radius: 16px; border: 1px solid #30363d;
    margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-box h1 { margin: 0 !important; font-size: 26px !important; }
.hero-box p  { color: #8b949e; font-size: 15px; margin: 6px 0 0 0; }

.stat-card {
    background: #161b22; padding: 20px 16px; border-radius: 12px;
    border: 1px solid #30363d; text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25); transition: transform .15s ease;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-value { font-size: 34px; font-weight: 700; line-height: 1; }
.stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase;
              letter-spacing: 1.4px; margin-top: 6px; }

.green  { color: #3fb950; } .red    { color: #f85149; }
.blue   { color: #58a6ff; } .gold   { color: #d29922; }
.purple { color: #bc8cff; }

.risk-box { padding: 22px 28px; border-radius: 14px; margin: 16px 0; border-left: 6px solid; }
.risk-low      { background: #0d2818; border-color: #3fb950; }
.risk-moderate { background: #2d2000; border-color: #d29922; }
.risk-high     { background: #300a0a; border-color: #f85149; }
.risk-critical { background: #3d0808; border-color: #ff4040; }
.risk-box h2   { margin: 0 0 6px 0 !important; font-size: 22px !important; }
.risk-box p    { margin: 0; font-size: 15px; color: #c9d1d9; }

.pipeline-step {
    background: #161b22; padding: 16px; border-radius: 10px;
    border: 1px solid #30363d; height: 100%;
}
.pipeline-step h4 { margin: 0 0 6px 0 !important; font-size: 14px !important; color: #58a6ff !important; }
.pipeline-step p  { margin: 0; color: #8b949e; font-size: 13px; }

[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: white; border: none; width: 100%; border-radius: 8px;
    font-weight: 600; padding: 0.6rem 1rem; font-size: 15px;
    box-shadow: 0 4px 12px rgba(35,134,54,0.3); transition: background .2s;
}
.stButton > button:hover { background: linear-gradient(135deg,#2ea043 0%,#3fb950 100%); }
hr { border-color: #21262d; margin: 20px 0; }
</style>
''', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def risk_config(pct):
    if pct < 5:
        return dict(level="LOW", css="risk-low", color="#3fb950", emoji="OK",
            summary="The patient's heartbeat pattern is predominantly normal.",
            detail=f"{pct:.1f}% of beats show arrhythmic patterns -- within the expected range for a healthy individual. No immediate cardiac intervention is indicated.",
            outlook="**Prognosis:** Low risk. Routine ECG follow-up every 12 months is recommended.")
    if pct < 20:
        return dict(level="MODERATE", css="risk-moderate", color="#d29922", emoji="WARNING",
            summary="A notable percentage of heartbeats show arrhythmic patterns.",
            detail=f"{pct:.1f}% of beats are flagged as arrhythmic. This suggests intermittent cardiac rhythm irregularities. Further diagnostic workup (Holter monitor, echocardiogram) is recommended.",
            outlook="**Prognosis:** Monitor every 3-6 months. Unmanaged arrhythmias may progress and elevate stroke or heart-failure risk.")
    if pct < 50:
        return dict(level="HIGH", css="risk-high", color="#f85149", emoji="HIGH RISK",
            summary="A significant portion of heartbeats exhibit arrhythmic activity.",
            detail=f"{pct:.1f}% of beats are classified as arrhythmic. This indicates persistent cardiac rhythm disturbance. Immediate cardiology consultation is strongly recommended.",
            outlook="**Prognosis:** Elevated risk of atrial fibrillation, ventricular tachycardia, stroke, and heart failure. Early treatment can significantly improve outcomes.")
    return dict(level="CRITICAL", css="risk-critical", color="#ff4040", emoji="CRITICAL",
        summary="The majority of heartbeats show arrhythmic patterns.",
        detail=f"{pct:.1f}% of beats are flagged as arrhythmic -- a severe, sustained cardiac rhythm disorder. Urgent cardiology referral is essential.",
        outlook="**Prognosis:** Critical risk. Associated with sudden cardiac arrest, cardiomyopathy, and thromboembolic events. Immediate intervention is necessary.")


def compute_performance_metrics(y_true, y_pred):
    """
    Computes robust classification performance metrics (Accuracy, Precision, Recall, F1)
    handling mixed datasets, pure normal (low-risk), and pure arrhythmia cases.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0, "No Data", "Precision", "Recall"

    acc = float(accuracy_score(y_true, y_pred))
    unique_true = np.unique(y_true)

    if len(unique_true) == 1 and unique_true[0] == 0:
        # All true beats are Normal (e.g. patient_low_risk.csv, patient_post_op_stable.csv)
        # Evaluates Normal rhythm recognition & specificity
        prec = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
        desc = "Normal Rhythm Recognition"
        prec_label = "Precision (Normal)"
        rec_label = "Specificity (Recall)"
    elif len(unique_true) == 1 and unique_true[0] == 1:
        # All true beats are Arrhythmia (e.g. acute pathology)
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        desc = "Arrhythmia Sensitivity Evaluation"
        prec_label = "Precision"
        rec_label = "Sensitivity (Recall)"
    else:
        # Mixed beats present (e.g. patient_high_risk.csv, general telemetry)
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        desc = "Arrhythmia Detection"
        prec_label = "Precision"
        rec_label = "Recall"

    return acc, prec, rec, f1, desc, prec_label, rec_label


def dark_fig(**kwargs):
    BG, PAN, GRD = "#0d1117", "#161b22", "#30363d"
    fig, ax = plt.subplots(**kwargs)
    fig.patch.set_facecolor(BG)
    axes = ax.flat if isinstance(ax, np.ndarray) else [ax]
    for a in axes:
        a.set_facecolor(PAN)
        for spine in a.spines.values(): spine.set_color(GRD)
        a.tick_params(colors="#e6edf3")
        a.grid(True, color=GRD, alpha=0.3)
    return fig, ax


def plot_transition_matrix(m_norm, n_states):
    labels = ["S0 Resting", "S1 P-Wave", "S2 QRS", "S3 T-Wave"][:n_states]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
    tm = m_norm.transmat_
    im = ax.imshow(tm, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    for i in range(n_states):
        for j in range(n_states):
            v = tm[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v > 0.5 else "#0d1117")
    ax.set_xticks(range(n_states)); ax.set_yticks(range(n_states))
    ax.set_xticklabels(labels, fontsize=9, color="#e6edf3", rotation=15)
    ax.set_yticklabels(labels, fontsize=9, color="#e6edf3")
    ax.set_xlabel("To State", fontsize=10, fontweight="bold", color="#e6edf3")
    ax.set_ylabel("From State", fontsize=10, fontweight="bold", color="#e6edf3")
    ax.set_title("Learned Transition Matrix -- Normal HMM",
                 fontsize=12, fontweight="bold", color="#58a6ff", pad=12)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Transition Probability")
    fig.tight_layout(); return fig


def plot_ll_histogram(lls, y_true=None):
    fig, ax = dark_fig(figsize=(10, 4), dpi=140)
    d = lls[:, 0] - lls[:, 1]
    if y_true is not None:
        ax.hist(d[y_true == 0], bins=80, alpha=0.65, color="#3fb950",
                label="True Normal", density=True, edgecolor="none")
        ax.hist(d[y_true == 1], bins=80, alpha=0.65, color="#f85149",
                label="True Arrhythmia", density=True, edgecolor="none")
        ax.legend(fontsize=10, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    else:
        ax.hist(d, bins=80, alpha=0.8, color="#58a6ff", density=True, edgecolor="none")
    ax.axvline(0, color="#58a6ff" if y_true is None else "#ffffff",
               ls="--", lw=1.8, label="Decision boundary")
    ax.set_xlabel("Delta Log-Likelihood (Normal LL - Arrhythmia LL)",
                  fontsize=11, fontweight="bold", color="#e6edf3")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold", color="#e6edf3")
    ax.set_title("Log-Likelihood Separation -- HMM Decision Boundary",
                 fontsize=13, fontweight="bold", color="#58a6ff", pad=10)
    fig.tight_layout(); return fig


def plot_ecg_waveform(beat_row, beat_idx, pred):
    color = "#3fb950" if pred == 0 else "#f85149"
    label = "Normal" if pred == 0 else "Arrhythmia"
    fig, ax = dark_fig(figsize=(10, 3), dpi=140)
    t = np.arange(len(beat_row)) / 125.0
    ax.plot(t, beat_row, color=color, lw=1.4)
    ax.fill_between(t, beat_row, alpha=0.12, color=color)
    ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold", color="#e6edf3")
    ax.set_ylabel("Voltage (norm.)", fontsize=10, fontweight="bold", color="#e6edf3")
    ax.set_title(f"Beat #{beat_idx + 1}  --  Predicted: {label}",fontsize=12, fontweight="bold", color=color, pad=8)
    fig.tight_layout(); return fig


def render_results(preds, lls, y_true, X_test, m_norm, n_states, elapsed, source):
    """Shared results section: cards + risk banner + tabs + ECG preview."""
    total  = len(preds)
    normal = int((preds == 0).sum())
    arrh   = int((preds == 1).sum())
    pct    = arrh / total * 100
    risk   = risk_config(pct)

    # -- Diagnostic summary cards
    st.markdown("### Diagnostic Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-value blue">{total:,}</div>'
                    f'<div class="stat-label">Total Heartbeats Evaluated</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-value red">{arrh:,}</div>'
                    f'<div class="stat-label">Detected Arrhythmias</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card">'
                    f'<div class="stat-value" style="color:{risk["color"]};">{pct:.1f}%</div>'
                    f'<div class="stat-label">Arrhythmic Burden</div></div>',unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Clinical risk banner
    st.markdown(
        f'<div class="risk-box {risk["css"]}">'
        f'<h2 style="color:{risk["color"]};">[{risk["emoji"]}] Risk Level: {risk["level"]}</h2>'
        f'<p>{risk["summary"]}</p></div>',unsafe_allow_html=True)

    with st.expander("Full Clinical Analysis", expanded=False):
        st.markdown(f"**Findings:** {risk['detail']}")
        st.markdown(risk['outlook'])
        st.markdown("---")
        st.caption("Disclaimer: This tool is for academic/research purposes only and is NOT a "
                   "substitute for professional medical diagnosis. Always consult a qualified "
                   "cardiologist for clinical decisions.")

    st.markdown("---")

    # -- Visualisation tabs
    st.markdown("### Visualisations")
    tab1, tab2 = st.tabs(["Transition Matrix", "Log-Likelihood Separation"])

    with tab1:
        st.markdown("The **4x4 Transition Matrix** of the Normal HMM encodes how the model "
                    "learned to move between the four cardiac sub-phases (Resting, P-Wave, "
                    "QRS complex, T-Wave) identified by PCA.")
        fig_tm = plot_transition_matrix(m_norm, n_states)
        st.pyplot(fig_tm); plt.close(fig_tm)

    with tab2:
        st.markdown("Each beat is scored by both HMMs. **Delta-LL > 0** -> Normal; ""**Delta-LL < 0** -> Arrhythmia. Wider separation = more confident predictions.")
        fig_ll = plot_ll_histogram(lls, y_true)
        st.pyplot(fig_ll); plt.close(fig_ll)

    st.markdown("---")

    # -- ECG waveform expander
    with st.expander("Raw ECG Waveform Preview", expanded=False):
        st.markdown("Select any beat index to inspect its raw 187-sample voltage trace.")
        max_idx = min(len(preds) - 1, 999)
        beat_idx = st.slider("Beat index", 0, max_idx, 0, key=f"slider_{source}")
        tag = "Normal" if preds[beat_idx] == 0 else "Arrhythmia"
        col_tag = "green" if preds[beat_idx] == 0 else "red"
        st.markdown(f'**Prediction:** <span class="{col_tag}">{tag}</span>',
                    unsafe_allow_html=True)
        fig_w = plot_ecg_waveform(X_test[beat_idx], beat_idx, preds[beat_idx])
        st.pyplot(fig_w); plt.close(fig_w)

    st.markdown("---")
    # -- Footer info row
    cols = st.columns(4)
    for col, (label, val) in zip(cols, [
        ("Inference Time",   f"{elapsed:.1f} s"),
        ("PCA Components",   "25 (93.4% var.)"),
        ("Covariance Type",  "Full (D x D)"),
        ("Source",           source),
    ]):
        with col:
            st.markdown(f'<div class="stat-card">'
                        f'<div class="stat-value blue" style="font-size:16px;">{val}</div>'
                        f'<div class="stat-label">{label}</div></div>',
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  CACHED MODEL TRAINING  (trains once on MIT-BIH, reused for upload inference)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training HMM models on MIT-BIH reference data...")
def get_trained_models(n_per_class, n_states, n_iter, n_components):
    df_tr = pd.read_csv(TRAIN_CSV, header=None)
    X_tr  = df_tr.iloc[:, :187].values
    y_tr  = (df_tr.iloc[:, 187].values.astype(int) > 0).astype(int)
    X0, l0, X1, l1, pca = prepare_hmm_data(
        X_tr, y_tr, n_per_class=n_per_class, n_components=n_components)
    m0, m1 = train_both(X0, l0, X1, l1,
                        n_states=n_states, n_iter=n_iter,
                        covariance_type="full")
    return m0, m1, pca


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## HMM Arrhythmia Pipeline")
    st.markdown("*Cardiac Diagnostic Engine*")
    st.markdown("---")

    mode = st.radio("**Analysis Mode**",
                    ["Upload ECG CSV", "Live Patient Streaming",
                     "Full MIT-BIH Benchmark"],
                    index=0)

    st.markdown("---")
    st.markdown("**Model Hyperparameters**")
    n_states     = st.slider("Hidden States",          2,    8,    6,   1)
    n_per_class  = st.slider("Training Samples/Class", 50,   3000, 200, 50)
    n_iter       = st.slider("Max EM Iterations",      10,   200,  100, 10)
    n_components = st.slider("PCA Components",         5,    50,   25,  5)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px; color:#8b949e; line-height:1.7;">'
        '<b>Algorithm:</b> Scaled Forward-Backward<br>'
        '<b>Covariance:</b> Full (D x D) per state<br>'
        '<b>Decoding:</b> Viterbi (log-space)<br>'
        '<b>Dataset:</b> MIT-BIH Arrhythmia DB (PhysioNet)</div>',
        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-box">'
    '<h1>Cardiac Arrhythmia Diagnostic Pipeline</h1>'
    '<p>Full-Covariance Gaussian HMM &nbsp;|&nbsp; '
    'PCA Noise Filtering (25 PCs, 93.4% variance) &nbsp;|&nbsp; '
    'Baum-Welch EM Training &nbsp;|&nbsp; '
    'Viterbi Decoding &nbsp;|&nbsp; MIT-BIH Arrhythmia Database</p>'
    '</div>',
    unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  MODE A -- UPLOAD ECG CSV
# ──────────────────────────────────────────────────────────────────────────────
if mode == "Upload ECG CSV":

    st.markdown("## Upload Patient ECG File")
    st.markdown(
        "Upload a **CSV file** with **187 voltage columns** per row (one row = one heartbeat). "
        "An optional 188th column may contain ground-truth integer labels "
        "(0 = Normal, 1-4 = Arrhythmia). "
        "File format matches the MIT-BIH Kaggle dataset."
    )

    pa, pb, pc, pd_ = st.columns(4)
    steps = [
        ("Ingest",   "Parse CSV and extract 187 voltage samples per heartbeat."),
        ("PCA",      f"Compress {187} raw features to {n_components} principal components."),
        ("HMM Score",f"Score vs. pre-trained {n_states}-state Full-Cov HMMs."),
        ("Report",   "Compute arrhythmic burden and output clinical risk classification."),
    ]
    for col, (title, desc) in zip([pa, pb, pc, pd_], steps):
        with col:
            st.markdown(f'<div class="pipeline-step"><h4>{title}</h4>'
                        f'<p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop or browse -- accepts .csv files",
        type=["csv"], accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is not None:
        try:
            with st.spinner("Reading CSV..."):
                df_up = pd.read_csv(uploaded, header=None)

            n_cols = df_up.shape[1]
            if n_cols < 187:
                st.error(f"File has only {n_cols} columns -- expected at least 187. "
                         "Please check your file format.")
                st.stop()

            X_up = df_up.iloc[:, :187].values.astype(np.float32)
            y_up = None
            if n_cols > 187:
                y_up = (df_up.iloc[:, -1].values.astype(int) > 0).astype(int)

            st.success(f"Loaded {X_up.shape[0]:,} heartbeats from '{uploaded.name}'"
                       + (" -- labels detected." if y_up is not None else " -- no label column."))

            with st.spinner("Loading pre-trained HMM models (cached after first run)..."):
                m_norm, m_arrh, pca_obj = get_trained_models(
                    n_per_class, n_states, n_iter, n_components)

            # Inference with progress bar
            n_beats = X_up.shape[0]
            bar = st.progress(0, text="Running Viterbi decoding...")
            t0 = _time.time()
            preds_up = np.zeros(n_beats, dtype=int)
            lls_up   = np.zeros((n_beats, 2))
            step = max(1, n_beats // 40)

            for i in range(n_beats):
                beat = pca_obj.transform(X_up[i:i+1])
                ll0 = m_norm.score(beat); ll1 = m_arrh.score(beat)
                lls_up[i] = [ll0, ll1]
                preds_up[i] = 0 if ll0 >= ll1 else 1
                if (i + 1) % step == 0:
                    bar.progress((i + 1) / n_beats,
                                 text=f"Decoded {i+1:,}/{n_beats:,} beats...")

            elapsed_up = _time.time() - t0
            bar.progress(1.0, text=f"Done -- {n_beats:,} beats in {elapsed_up:.1f}s")

            st.markdown("---")
            render_results(preds_up, lls_up, y_up, X_up,
                           m_norm, n_states, elapsed_up, uploaded.name)

            # Optional accuracy metrics if labels were supplied
            if y_up is not None:
                acc_up, prec_up, rec_up, f1_up, desc_up, p_lbl, r_lbl = compute_performance_metrics(y_up, preds_up)
                st.markdown(f"### Classification Performance ({desc_up})")
                m1c, m2c, m3c, m4c = st.columns(4)
                for col, (name, val, cls) in zip(
                    [m1c, m2c, m3c, m4c],
                    [("Accuracy", acc_up,  "blue"),
                     (p_lbl,      prec_up, "gold"),
                     (r_lbl,      rec_up,  "green"),
                     ("F1-Score", f1_up,   "purple")],
                ):
                    with col:
                        st.markdown(
                            f'<div class="stat-card">'
                            f'<div class="stat-value {cls}">{val:.4f}</div>'
                            f'<div class="stat-label">{name}</div></div>',
                            unsafe_allow_html=True)

                with st.expander("Full Classification Report & Confusion Matrix", expanded=False):
                    report_str = classification_report(
                        y_up, preds_up, target_names=["Normal", "Arrhythmia"],
                        digits=4, zero_division=0
                    )
                    cm = confusion_matrix(y_up, preds_up)
                    st.code(f"Classification Report:\n{report_str}\n\nConfusion Matrix:\n{cm}")

        except Exception as exc:
            st.error(f"Error processing file: {exc}")
            st.exception(exc)
    else:
        st.markdown(
            '<div style="text-align:center; padding:60px 20px; border:2px dashed #30363d; '
            'border-radius:16px; background:#161b22; margin-top:16px;">'
            '<div style="font-size:52px;">&#128196;</div>'
            '<h3 style="color:#58a6ff; margin:12px 0 6px;">No file uploaded yet</h3>'
            '<p style="color:#8b949e; font-size:14px;">'
            'Upload a CSV with 187 voltage columns per row.<br>'
            'The model will run PCA and HMM inference and generate your clinical report.</p>'
            '</div>',
            unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  MODE B -- FULL MIT-BIH BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────
elif mode == "Full MIT-BIH Benchmark":
    st.markdown("## Full MIT-BIH Arrhythmia Benchmark")
    st.markdown(
        "Train both HMMs on mitbih_train.csv and evaluate on mitbih_test.csv "
        "with the hyperparameters set in the sidebar."
    )

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(f"Config: {n_states} states | {n_per_class} beats/class | "
                f"{n_iter} EM iters | {n_components} PCA components | Full covariance")
    with col_btn:
        run_btn = st.button("Run Full Pipeline")

    if not run_btn:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        for col, (val, label, cls) in zip([c1, c2, c3, c4], [
            ("87,554", "Training Heartbeats", "blue"),
            ("21,892", "Test Heartbeats",     "blue"),
            ("187",    "Samples / Beat",      "purple"),
            ("5",      "Original ECG Classes","gold"),
        ]):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value {cls}">{val}</div>' 
                            f'<div class="stat-label">{label}</div></div>',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("> Adjust hyperparameters in the sidebar, then click **Run Full Pipeline**.")
    else:
        with st.spinner("Running full HMM pipeline -- training and evaluating on MIT-BIH..."):
            result = run_pipeline(
                n_per_class=n_per_class, n_states=n_states,
                n_iter=n_iter, n_components=n_components,
                covariance_type="full", return_figs=False,
            )

        preds   = result["preds"]; lls = result["lls"]
        y_te    = result["y_true"]; X_te = result["X_test"]
        metrics = result["metrics"]; elapsed = result["elapsed"]

        # Fetch the warm cached models for transition matrix
        m_norm, _, _ = get_trained_models(n_per_class, n_states, n_iter, n_components)

        # Dataset overview
        st.markdown("### Dataset Overview")
        d1, d2, d3, d4 = st.columns(4)
        for col, (val, label, cls) in zip([d1, d2, d3, d4], [
            (f"{result['n_train']:,}", "Training Heartbeats", "blue"),
            (f"{result['n_test']:,}",  "Test Heartbeats",     "blue"),
            ("187",                    "Features / Beat",     "purple"),
            (f"{n_components}",        "PCA Components",      "gold"),
        ]):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value {cls}">{val}</div>'
                            f'<div class="stat-label">{label}</div></div>',
                            unsafe_allow_html=True)

        st.markdown("---")

        # All four metrics
        st.markdown("### Classification Performance")
        m1c, m2c, m3c, m4c = st.columns(4)
        for col, (name, val, cls) in zip([m1c, m2c, m3c, m4c], [
            ("Accuracy",  metrics["Accuracy"],  "blue"),
            ("Precision", metrics["Precision"], "gold"),
            ("Recall",    metrics["Recall"],    "green"),
            ("F1-Score",  metrics["F1-Score"],  "purple"),
        ]):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-value {cls}">{val:.4f}</div>'
                            f'<div class="stat-label">{name}</div></div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        render_results(preds, lls, y_te, X_te,
                       m_norm, n_states, elapsed, "mitbih_test.csv")

        with st.expander("Full Class Distribution", expanded=False):
            names = {0:"Normal",1:"Supraventricular",2:"Ventricular",3:"Fusion",4:"Unknown"}
            col_tr, col_te = st.columns(2)
            with col_tr:
                st.markdown("**Training Set**")
                st.dataframe(pd.DataFrame({
                    "Class": [f"{k} - {v}" for k,v in names.items()],
                    "Count": [int((result["y_train_raw"]==k).sum()) for k in range(5)],
                }), use_container_width=True, hide_index=True)
            with col_te:
                st.markdown("**Test Set**")
                st.dataframe(pd.DataFrame({
                    "Class": [f"{k} - {v}" for k,v in names.items()],
                    "Count": [int((result["y_test_raw"]==k).sum()) for k in range(5)],
                }), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
#  MODE C -- LIVE PATIENT STREAMING (REAL-TIME DEMO)
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("## Live Patient ECG Streaming")
    st.markdown(
        "Simulate a **real-time bedside ECG monitor** by streaming heartbeats "
        "one-by-one from a pre-generated virtual patient file. Each beat is "
        "scored live by the HMM and the clinical risk banner updates dynamically."
    )

    # -- Discover patient CSV files in the project directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    patient_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "patient_*.csv")))
    patient_names = [os.path.basename(f) for f in patient_files]

    if not patient_names:
        st.warning("No patient files found. Run `create_patients.py` first to generate "
                   "`patient_low_risk.csv` and `patient_high_risk.csv`.")
        st.stop()

    # -- Controls
    col_sel, col_spd = st.columns([3, 1])
    with col_sel:
        selected_file = st.selectbox(
            "Select Patient File", patient_names,
            index=0, key="stream_patient_select")
    with col_spd:
        beat_delay = st.number_input(
            "Delay per beat (s)", min_value=0.05, max_value=2.0,
            value=0.4, step=0.05, key="stream_delay",
            help="Target interval between beats (e.g., 0.80s = 75 BPM resting, 0.50s = 120 BPM, 0.40s = 150 BPM tachycardia).")

    start_btn = st.button("Start Live ECG Stream", key="stream_start",
                          use_container_width=True)

    st.markdown("---")

    # -- Load patient data
    csv_path = os.path.join(SCRIPT_DIR, selected_file)
    df_patient = pd.read_csv(csv_path, header=None)
    X_patient = df_patient.iloc[:, :187].values.astype(np.float32)
    n_beats = X_patient.shape[0]
    has_labels = df_patient.shape[1] > 187
    y_patient = (df_patient.iloc[:, -1].values.astype(int) > 0).astype(int) if has_labels else None

    # -- Load cached models
    with st.spinner("Loading pre-trained HMM models (cached)..."):
        m_norm, m_arrh, pca_obj = get_trained_models(
            n_per_class, n_states, n_iter, n_components)

    # Pre-score patient beats for instant display across all patient options
    X_pca_patient = pca_obj.transform(X_patient)
    preds_patient = np.zeros(n_beats, dtype=int)
    lls_patient = np.zeros((n_beats, 2))
    for b in range(n_beats):
        ll0 = m_norm.score(X_pca_patient[b:b+1])
        ll1 = m_arrh.score(X_pca_patient[b:b+1])
        lls_patient[b] = [ll0, ll1]
        preds_patient[b] = 0 if ll0 >= ll1 else 1

    arrh_cnt = int(preds_patient.sum())
    norm_cnt = n_beats - arrh_cnt
    burden_pct = (arrh_cnt / n_beats) * 100
    risk_info = risk_config(burden_pct)

    if has_labels:
        acc_p, prec_p, rec_p, f1_p, desc_p, p_label, r_label = compute_performance_metrics(y_patient, preds_patient)
    else:
        acc_p, prec_p, rec_p, f1_p, desc_p, p_label, r_label = 0.0, 0.0, 0.0, 0.0, "Telemetry Only", "Precision", "Recall"

    if not start_btn:
        st.markdown(f"### Live Patient Monitor: `{selected_file}`")

        # 1. Patient Bedside Telemetry
        st.markdown("#### Patient Bedside Telemetry")
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value blue">{n_beats}</div>'
                f'<div class="stat-label">Total Heartbeats</div></div>',
                unsafe_allow_html=True)
        with tc2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value green">{norm_cnt}</div>'
                f'<div class="stat-label">Normal Heartbeats</div></div>',
                unsafe_allow_html=True)
        with tc3:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value red">{arrh_cnt}</div>'
                f'<div class="stat-label">Detected Arrhythmias</div></div>',
                unsafe_allow_html=True)
        with tc4:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value" style="color:{risk_info["color"]};">{burden_pct:.1f}%</div>'
                f'<div class="stat-label">Arrhythmic Burden</div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Performance Metrics
        st.markdown(f"#### HMM Classification Performance ({desc_p})")
        if has_labels:
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value blue">{acc_p:.4f}</div>'
                    f'<div class="stat-label">Accuracy</div></div>',
                    unsafe_allow_html=True)
            with mc2:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value gold">{prec_p:.4f}</div>'
                    f'<div class="stat-label">{p_label}</div></div>',
                    unsafe_allow_html=True)
            with mc3:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value green">{rec_p:.4f}</div>'
                    f'<div class="stat-label">{r_label}</div></div>',
                    unsafe_allow_html=True)
            with mc4:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value purple">{f1_p:.4f}</div>'
                    f'<div class="stat-label">F1-Score</div></div>',
                    unsafe_allow_html=True)
        else:
            st.info("No ground-truth labels found in this patient record.")

        # 3. Clinical Risk Banner
        st.markdown(
            f'<div class="risk-box {risk_info["css"]}">'
            f'<h2 style="color:{risk_info["color"]};">[{risk_info["emoji"]}] Risk Level: {risk_info["level"]}</h2>'
            f'<p>{risk_info["summary"]}</p></div>',
            unsafe_allow_html=True)

        # 4. ECG Preview Waveform
        fig_prev, ax_prev = dark_fig(figsize=(12, 3), dpi=100)
        trace_prev = np.concatenate([X_patient[i] for i in range(min(5, n_beats))])
        t_axis_prev = np.arange(len(trace_prev)) / 125.0
        color_prev = "#3fb950" if preds_patient[0] == 0 else "#f85149"
        ax_prev.plot(t_axis_prev, trace_prev, color=color_prev, lw=1.0)
        ax_prev.fill_between(t_axis_prev, trace_prev, alpha=0.1, color=color_prev)
        ax_prev.set_title(f"ECG Trace Preview -- Initial 5 Heartbeats ({selected_file})",
                          fontsize=11, fontweight="bold", color="#58a6ff", pad=8)
        ax_prev.set_xlabel("Time (s)", fontsize=9, color="#e6edf3")
        ax_prev.set_ylabel("Voltage", fontsize=9, color="#e6edf3")
        fig_prev.tight_layout()
        st.pyplot(fig_prev)
        plt.close(fig_prev)

        st.caption("Click **Start Live ECG Stream** above to stream heartbeats one-by-one with real-time waveform animation.")
    else:
        # -- Real-time streaming simulation
        # Create live-updating placeholders
        ph_telemetry = st.empty()   # Bedside patient telemetry cards
        ph_perf      = st.empty()   # 4 classification performance metrics
        ph_risk      = st.empty()   # Risk banner
        ph_ecg       = st.empty()   # ECG waveform chart
        ph_status    = st.empty()   # Status text

        # Accumulators
        preds_all = []
        lls_all   = []
        ecg_window = []             # rolling window of recent voltages
        WINDOW = 5                  # show last N beats in the trace
        t0 = _time.time()

        for i in range(n_beats):
            t_beat_start = _time.time()
            beat_raw = X_patient[i]
            beat_pca = pca_obj.transform(beat_raw.reshape(1, -1))
            ll0 = m_norm.score(beat_pca)
            ll1 = m_arrh.score(beat_pca)
            pred = 0 if ll0 >= ll1 else 1
            preds_all.append(pred)
            lls_all.append([ll0, ll1])

            # Running stats
            total_so_far = len(preds_all)
            arrh_so_far  = sum(preds_all)
            norm_so_far  = total_so_far - arrh_so_far
            burden_pct_live = arrh_so_far / total_so_far * 100
            risk_live = risk_config(burden_pct_live)

            # Update Bedside Telemetry cards
            with ph_telemetry.container():
                st.markdown("#### Patient Bedside Telemetry")
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value blue">{total_so_far}</div>'
                        f'<div class="stat-label">Beats Processed</div></div>',
                        unsafe_allow_html=True)
                with mc2:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value green">{norm_so_far}</div>'
                        f'<div class="stat-label">Normal</div></div>',
                        unsafe_allow_html=True)
                with mc3:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value red">{arrh_so_far}</div>'
                        f'<div class="stat-label">Arrhythmias</div></div>',
                        unsafe_allow_html=True)
                with mc4:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div class="stat-value" style="color:{risk_live["color"]};">{burden_pct_live:.1f}%</div>'
                        f'<div class="stat-label">Arrhythmic Burden</div></div>',
                        unsafe_allow_html=True)

            # Update 4 Live Performance Metrics
            with ph_perf.container():
                if has_labels:
                    y_true_so_far = y_patient[:total_so_far]
                    acc_val, prec_val, rec_val, f1_val, desc_val, p_lbl_live, r_lbl_live = compute_performance_metrics(
                        y_true_so_far, preds_all)

                    st.markdown(f"#### Live HMM Classification Performance ({desc_val})")
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    with pc1:
                        st.markdown(
                            f'<div class="stat-card"><div class="stat-value blue">{acc_val:.4f}</div>'
                            f'<div class="stat-label">Accuracy</div></div>',
                            unsafe_allow_html=True)
                    with pc2:
                        st.markdown(
                            f'<div class="stat-card"><div class="stat-value gold">{prec_val:.4f}</div>'
                            f'<div class="stat-label">{p_lbl_live}</div></div>',
                            unsafe_allow_html=True)
                    with pc3:
                        st.markdown(
                            f'<div class="stat-card"><div class="stat-value green">{rec_val:.4f}</div>'
                            f'<div class="stat-label">{r_lbl_live}</div></div>',
                            unsafe_allow_html=True)
                    with pc4:
                        st.markdown(
                            f'<div class="stat-card"><div class="stat-value purple">{f1_val:.4f}</div>'
                            f'<div class="stat-label">F1-Score</div></div>',
                            unsafe_allow_html=True)
                else:
                    st.info("No ground-truth labels in this file -- live evaluation metrics require labeled heartbeats.")

            # Update risk banner
            ph_risk.markdown(
                f'<div class="risk-box {risk_live["css"]}">'
                f'<h2 style="color:{risk_live["color"]};">[{risk_live["emoji"]}] Risk Level: {risk_live["level"]}</h2>'
                f'<p>{risk_live["summary"]}</p></div>',
                unsafe_allow_html=True)

            # Update ECG waveform (rolling window)
            ecg_window.append(beat_raw)
            if len(ecg_window) > WINDOW:
                ecg_window.pop(0)

            with ph_ecg.container():
                trace = np.concatenate(ecg_window)
                fig_live, ax_live = dark_fig(figsize=(12, 3), dpi=100)
                t_axis = np.arange(len(trace)) / 125.0
                color = "#3fb950" if pred == 0 else "#f85149"
                ax_live.plot(t_axis, trace, color=color, lw=0.9)
                ax_live.fill_between(t_axis, trace, alpha=0.08, color=color)
                label = "Normal" if pred == 0 else "ARRHYTHMIA"
                ax_live.set_title(
                    f"Live ECG  |  Beat #{i+1}/{n_beats}  |  Latest: {label}",
                    fontsize=12, fontweight="bold", color=color, pad=8)
                ax_live.set_xlabel("Time (s)", fontsize=9, color="#e6edf3")
                ax_live.set_ylabel("Voltage", fontsize=9, color="#e6edf3")
                # Mark beat boundaries
                for b in range(1, len(ecg_window)):
                    bx = b * 187 / 125.0
                    ax_live.axvline(bx, color="#30363d", ls=":", lw=0.7)
                fig_live.tight_layout()
                st.pyplot(fig_live)
                plt.close(fig_live)

            # Status line with BPM indicator
            elapsed_so_far = _time.time() - t0
            ms_per_beat = (elapsed_so_far / total_so_far) * 1000
            curr_bpm = 60000.0 / max(1.0, ms_per_beat)
            ph_status.markdown(
                f'<div style="text-align:center; color:#8b949e; font-size:13px;">'
                f'Elapsed: {elapsed_so_far:.1f}s  |  '
                f'Cadence: {ms_per_beat:.0f} ms/beat (~{curr_bpm:.0f} BPM)  |  '
                f'Progress: {total_so_far}/{n_beats}'
                f'</div>',
                unsafe_allow_html=True)

            # Latency compensation: sleep only the remaining time after computation & rendering
            t_work = _time.time() - t_beat_start
            sleep_duration = max(0.005, beat_delay - t_work)
            _time.sleep(sleep_duration)

        # ── Stream complete ── show final summary
        total_elapsed = _time.time() - t0
        net_inference = total_elapsed - (n_beats * beat_delay)  # subtract sleep time
        preds_arr = np.array(preds_all)
        lls_arr   = np.array(lls_all)
        final_burden = sum(preds_all) / n_beats * 100
        final_risk = risk_config(final_burden)

        st.markdown("---")
        st.markdown("### Stream Complete")

        fc1, fc2, fc3, fc4 = st.columns(4)
        for col, (val, label, cls) in zip([fc1, fc2, fc3, fc4], [
            (f"{n_beats}",                          "Total Beats Streamed", "blue"),
            (f"{net_inference*1000/n_beats:.2f} ms", "Net Inference / Beat", "green"),
            (f"{total_elapsed:.1f} s",              "Total Wall Time",     "gold"),
            (f"{final_burden:.1f}%",                "Final Burden",         "red"),
        ]):
            with col:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-value {cls}" style="font-size:20px;">{val}</div>'
                    f'<div class="stat-label">{label}</div></div>',
                    unsafe_allow_html=True)

        # Show detailed final report if labels exist
        if y_patient is not None:
            with st.expander("Full Classification Report & Confusion Matrix", expanded=True):
                report_str = classification_report(
                    y_patient, preds_arr, labels=[0, 1], target_names=["Normal", "Arrhythmia"],
                    digits=4, zero_division=0
                )
                cm = confusion_matrix(y_patient, preds_arr, labels=[0, 1])
                st.code(f"Classification Report:\n{report_str}\n\nConfusion Matrix (Rows: True [Norm, Arrh], Cols: Pred [Norm, Arrh]):\n{cm}")

        # Final risk summary
        st.markdown("---")
        st.markdown(
            f'<div class="risk-box {final_risk["css"]}">'
            f'<h2 style="color:{final_risk["color"]};">[{final_risk["emoji"]}] '
            f'Final Diagnosis: {final_risk["level"]} RISK</h2>'
            f'<p>{final_risk["detail"]}</p></div>',
            unsafe_allow_html=True)

        with st.expander("Full Clinical Analysis", expanded=False):
            st.markdown(f"**Findings:** {final_risk['detail']}")
            st.markdown(final_risk['outlook'])
            st.markdown("---")
            st.caption("Disclaimer: This tool is for academic/research purposes only and is NOT a "
                       "substitute for professional medical diagnosis. Always consult a qualified "
                       "cardiologist for clinical decisions.")
