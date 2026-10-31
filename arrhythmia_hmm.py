"""
================================================================================
 PREDICTIVE ARRHYTHMIA DETECTION VIA HIDDEN MARKOV MODELS
 Domain:  Biomedical Signal Processing & Machine Learning
 Dataset: MIT-BIH Arrhythmia Database (Kaggle CSV format)
 Engine:  Custom Gaussian HMM (Baum-Welch + Viterbi from scratch)
================================================================================
 Phase 1 -- Data Ingestion & Feature Engineering
            ** PCA preprocessing: 187 raw voltage features -> n_components PCs
               to strip high-frequency sensor noise before HMM training.
 Phase 2 -- HMM Training (Baum-Welch / Expectation-Maximization)
            ** covariance_type='full' captures cross-feature correlations.
            ** n_iter increased to 100 for cleaner EM convergence.
 Phase 3 -- Real-Time Decoding & Prediction (Viterbi Algorithm)
 Phase 4 -- Performance Dashboard & Visualization
================================================================================
 The GaussianHMM uses SCALED forward/backward (not log-space).
 Each forward step is a numpy dot product: alpha[t] = (alpha[t-1] @ A) * B[t]
 This is orders of magnitude faster than logsumexp-based implementations.

 Covariance modes:
   'diag' -- diagonal covariance (original, fast, features treated independently)
   'full' -- full covariance matrix per state (captures cross-PC correlations)
================================================================================
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(SCRIPT_DIR, "mitbih_train.csv")
TEST_CSV   = os.path.join(SCRIPT_DIR, "mitbih_test.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def flush_print(*args, **kwargs):
    """Print with immediate flush so output appears in real-time."""
    print(*args, **kwargs, flush=True)


# ============================================================================
#  GAUSSIAN HMM -- SCALED FORWARD/BACKWARD (FAST)
# ============================================================================

class GaussianHMM:
    """
    Gaussian HMM with selectable covariance type ('diag' or 'full').

    Uses SCALED forward/backward algorithm (not log-space).
    Key insight: each forward step is just alpha @ A * B[t], which is a
    fast numpy dot product -- no logsumexp overhead.

    covariance_type='diag' -- original diagonal mode (K, D) variances.
    covariance_type='full' -- full (K, D, D) covariance matrices; captures
                              cross-feature correlations across PCA components.

    Training: Baum-Welch (EM)
    Decoding: Viterbi algorithm
    Scoring:  Forward algorithm (returns log-likelihood)
    """

    def __init__(self, n_states=4, n_iter=100, tol=1e-3, random_state=42,
                 covariance_type="full"):
        if covariance_type not in ("diag", "full"):
            raise ValueError("covariance_type must be 'diag' or 'full'")
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.rng = np.random.default_rng(random_state)
        self.pi = None      # (K,)
        self.A = None       # (K, K) transition matrix
        self.means = None   # (K, D)
        self.covars = None  # (K, D) diag variances  OR  (K, D, D) full matrices

    def _init_params(self, X_all, lengths):
        K, D = self.K, X_all.shape[1]
        self.pi = np.ones(K) / K
        self.A = np.ones((K, K)) / K + self.rng.uniform(0, 0.1, (K, K))
        self.A /= self.A.sum(1, keepdims=True)
        # Quantile-based mean init along first feature
        sv = np.sort(X_all[:, 0])
        qi = np.linspace(0, len(sv)-1, K+1).astype(int)
        self.means = np.zeros((K, D))
        for k in range(K):
            seg = X_all[qi[k]:qi[k+1]]
            self.means[k] = seg.mean(0) if len(seg) > 0 else 0.0

        if self.covariance_type == "diag":
            # (K, D) diagonal variances
            self.covars = np.tile(np.maximum(np.var(X_all, 0), 1e-4), (K, 1))
        else:
            # (K, D, D) full covariance -- initialise as scaled identity
            global_var = np.maximum(np.var(X_all, 0).mean(), 1e-4)
            self.covars = np.stack([global_var * np.eye(D)] * K)  # (K, D, D)

    # ------------------------------------------------------------------
    #  Emission probability helpers
    # ------------------------------------------------------------------

    def _emission_prob_diag(self, X):
        """Diagonal-covariance emission: B[t, k] = N(x_t | mu_k, diag(var_k))."""
        T, D = X.shape
        K = self.K
        B = np.zeros((T, K))
        for k in range(K):
            diff = X - self.means[k]          # (T, D)
            var_k = self.covars[k]             # (D,)
            exponent = -0.5 * np.sum(diff**2 / var_k, axis=1)
            norm = np.prod(np.sqrt(2 * np.pi * var_k))
            B[:, k] = np.exp(exponent) / norm
        return np.maximum(B, 1e-300)

    def _emission_prob_full(self, X):
        """Full-covariance emission: B[t, k] = N(x_t | mu_k, Sigma_k)."""
        T, D = X.shape
        K = self.K
        B = np.zeros((T, K))
        for k in range(K):
            Sigma = self.covars[k]             # (D, D)
            try:
                # Cholesky for numerical stability
                L = np.linalg.cholesky(Sigma)  # lower-triangular
                diff = X - self.means[k]       # (T, D)
                # Solve L @ z = diff.T  ->  z = L^{-1} diff.T
                z = np.linalg.solve(L, diff.T) # (D, T)
                maha = np.sum(z**2, axis=0)    # (T,) Mahalanobis^2
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                log_norm = 0.5 * (D * np.log(2 * np.pi) + log_det)
                B[:, k] = np.exp(-0.5 * maha - log_norm)
            except np.linalg.LinAlgError:
                # Fallback: diagonal approximation if Sigma is not PD
                var_k = np.maximum(np.diag(Sigma), 1e-4)
                diff = X - self.means[k]
                exponent = -0.5 * np.sum(diff**2 / var_k, axis=1)
                norm = np.prod(np.sqrt(2 * np.pi * var_k))
                B[:, k] = np.exp(exponent) / norm
        return np.maximum(B, 1e-300)

    def _emission_prob(self, X):
        """Dispatch to the correct emission helper based on covariance_type."""
        if self.covariance_type == "full":
            return self._emission_prob_full(X)
        return self._emission_prob_diag(X)

    def _forward_scaled(self, B):
        """
        Scaled forward algorithm.
        Returns alpha_hat (T, K) and scale factors c (T,).
        log P(O|lambda) = sum(log(c))
        """
        T, K = B.shape
        alpha = np.zeros((T, K))
        c = np.zeros(T)

        # t=0
        alpha[0] = self.pi * B[0]
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0]
        else:
            alpha[0] = 1.0 / K
            c[0] = 1e-300

        # t=1..T-1
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t]
            else:
                alpha[t] = 1.0 / K
                c[t] = 1e-300

        return alpha, c

    def _backward_scaled(self, B, c):
        """Scaled backward algorithm. Returns beta_hat (T, K)."""
        T, K = B.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0

        for t in range(T-2, -1, -1):
            beta[t] = (self.A @ (B[t+1] * beta[t+1]))
            if c[t+1] > 0:
                beta[t] /= c[t+1]

        return beta

    def _compute_xi_gamma(self, B, alpha, beta, c):
        """
        Compute gamma (T,K) and xi_sum (K,K).
        xi_sum accumulates transition counts without storing the full (T,K,K) tensor.
        """
        T, K = B.shape
        gamma = alpha * beta
        g_sum = gamma.sum(1, keepdims=True)
        g_sum = np.maximum(g_sum, 1e-300)
        gamma /= g_sum

        # Accumulate xi directly (avoids storing T x K x K tensor)
        xi_sum = np.zeros((K, K))
        for t in range(T-1):
            numer = np.outer(alpha[t], B[t+1] * beta[t+1]) * self.A
            denom = numer.sum()
            if denom > 0:
                xi_sum += numer / denom

        return gamma, xi_sum

    def fit(self, X_all, lengths):
        """Train via Baum-Welch EM."""
        self._init_params(X_all, lengths)

        # Split into sequences
        seqs = []
        idx = 0
        for L in lengths:
            seqs.append(X_all[idx:idx+L])
            idx += L

        K, D = self.K, X_all.shape[1]
        prev_ll = -np.inf

        for iteration in range(1, self.n_iter + 1):
            total_ll = 0.0

            # Accumulators
            gamma_init_sum = np.zeros(K)
            xi_total = np.zeros((K, K))
            gamma_total = np.zeros(K)
            mean_num = np.zeros((K, D))
            # Covariance accumulators: (K, D) for diag, (K, D, D) for full
            if self.covariance_type == "full":
                cov_num = np.zeros((K, D, D))
            else:
                var_num = np.zeros((K, D))

            for X in seqs:
                B = self._emission_prob(X)
                alpha, c = self._forward_scaled(B)
                beta = self._backward_scaled(B, c)
                gamma, xi_sum = self._compute_xi_gamma(B, alpha, beta, c)

                total_ll += np.sum(np.log(np.maximum(c, 1e-300)))

                gamma_init_sum += gamma[0]
                xi_total += xi_sum
                gk = gamma.sum(0)
                gamma_total += gk
                mean_num += gamma.T @ X
                for k in range(K):
                    diff = X - self.means[k]          # (T, D)
                    if self.covariance_type == "full":
                        # Weighted outer-product accumulation: sum_t gamma[t,k] * diff[t] diff[t]^T
                        cov_num[k] += (gamma[:, k, None, None] *
                                       diff[:, :, None] * diff[:, None, :]).sum(0)
                    else:
                        var_num[k] += (gamma[:, k:k+1] * diff**2).sum(0)

            # M-step: update parameters
            self.pi = gamma_init_sum / gamma_init_sum.sum()
            self.pi = np.maximum(self.pi, 1e-10)

            for j in range(K):
                row_sum = xi_total[j].sum()
                if row_sum > 0:
                    self.A[j] = xi_total[j] / row_sum
                self.A[j] = np.maximum(self.A[j], 1e-10)
            self.A /= self.A.sum(1, keepdims=True)

            for k in range(K):
                if gamma_total[k] > 1e-10:
                    self.means[k] = mean_num[k] / gamma_total[k]
                    if self.covariance_type == "full":
                        Sigma = cov_num[k] / gamma_total[k]
                        # Regularise: add small ridge to guarantee PD
                        Sigma += 1e-4 * np.eye(D)
                        self.covars[k] = Sigma
                    else:
                        self.covars[k] = var_num[k] / gamma_total[k]

            if self.covariance_type == "diag":
                self.covars = np.maximum(self.covars, 1e-4)

            improvement = total_ll - prev_ll
            if iteration % 5 == 0 or iteration == 1:
                flush_print(f"      EM iter {iteration:>3d} | "
                            f"LL = {total_ll:>12.2f} | "
                            f"delta = {improvement:+.2f}")

            if abs(improvement) < self.tol and iteration > 2:
                flush_print(f"      >> Converged at iteration {iteration}")
                break
            prev_ll = total_ll

        return self

    def score(self, X):
        """Log P(O | lambda) via scaled forward algorithm."""
        B = self._emission_prob(X)
        _, c = self._forward_scaled(B)
        return np.sum(np.log(np.maximum(c, 1e-300)))

    def predict(self, X):
        """Viterbi decoding (log-space for numerical stability)."""
        T, K = X.shape[0], self.K
        B = self._emission_prob(X)
        log_B = np.log(np.maximum(B, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_pi = np.log(np.maximum(self.pi, 1e-300))

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            tmp = delta[t-1, :, None] + log_A  # (K, K)
            psi[t] = tmp.argmax(axis=0)
            delta[t] = tmp.max(axis=0) + log_B[t]

        states = np.zeros(T, dtype=int)
        states[-1] = delta[-1].argmax()
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    @property
    def transmat_(self):
        return self.A

    @property
    def means_(self):
        return self.means


# ============================================================================
#  PHASE 1 -- DATA INGESTION
# ============================================================================

def load_data(train_path, test_path):
    flush_print("=" * 72)
    flush_print("  PHASE 1: DATA INGESTION & FEATURE ENGINEERING")
    flush_print("=" * 72)
    t0 = time.time()
    df_train = pd.read_csv(train_path, header=None)
    df_test  = pd.read_csv(test_path, header=None)
    X_train = df_train.iloc[:, :187].values
    y_train = df_train.iloc[:, 187].values.astype(int)
    X_test  = df_test.iloc[:, :187].values
    y_test  = df_test.iloc[:, 187].values.astype(int)
    flush_print(f"  Train: {X_train.shape[0]:,} beats x {X_train.shape[1]} steps")
    flush_print(f"  Test : {X_test.shape[0]:,} beats x {X_test.shape[1]} steps")
    flush_print(f"  Loaded in {time.time()-t0:.1f}s")
    return X_train, y_train, X_test, y_test


def binarize_labels(y):
    y_bin = (y > 0).astype(int)
    n0, n1 = (y_bin==0).sum(), (y_bin==1).sum()
    flush_print(f"  Binary: Normal={n0:,} | Arrhythmia={n1:,} | Ratio={n0/max(n1,1):.1f}:1")
    return y_bin


def prepare_hmm_data(X, y_bin, n_per_class=200, random_state=42,
                     n_components=25):
    """
    Downsample + PCA compress + concatenate beats into ONE sequence per class.

    PCA step
    --------
    Fit PCA on the combined sample of both classes (n_per_class * 2 beats),
    then project every beat from 187 raw voltage features down to
    ``n_components`` principal components.  This strips high-frequency sensor
    noise while preserving the dominant cardiac morphology variance.

    The resulting sequences have shape (n_beats * n_components,) treated as
    a multi-dimensional time series with D = n_components features per step
    where each "step" corresponds to one beat.
    """
    rng = np.random.default_rng(random_state)
    i0 = rng.choice(np.where(y_bin==0)[0], min(n_per_class, (y_bin==0).sum()), replace=False)
    i1 = rng.choice(np.where(y_bin==1)[0], min(n_per_class, (y_bin==1).sum()), replace=False)

    # ---- PCA: fit on training sample, apply to all ----
    flush_print(f"\n  PCA preprocessing: 187 raw features -> {n_components} principal components")
    pca_sample = np.concatenate([X[i0], X[i1]], axis=0)  # (2*n_per_class, 187)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(pca_sample)
    explained = pca.explained_variance_ratio_.sum() * 100
    flush_print(f"  PCA fitted | variance explained by top {n_components} PCs: {explained:.1f}%")

    # Project selected beats (shape: n_beats x n_components)
    X0_pca = pca.transform(X[i0])   # (n0, n_components)
    X1_pca = pca.transform(X[i1])   # (n1, n_components)

    # Each class: ONE long concatenated sequence of shape (n*n_components,)
    # Treat each beat as a single timestep with n_components-dimensional observation
    flush_print(f"\n  HMM training data ({n_per_class} beats/class):")
    flush_print(f"  Normal    : {len(i0)} beats -> {X0_pca.shape} (beats x PCs)")
    flush_print(f"  Arrhythmia: {len(i1)} beats -> {X1_pca.shape} (beats x PCs)")

    # lengths: [n_beats] per class -- each beat is ONE timestep
    return X0_pca, [X0_pca.shape[0]], X1_pca, [X1_pca.shape[0]], pca


# ============================================================================
#  PHASE 2 -- HMM TRAINING
# ============================================================================

def train_hmm(X_concat, lengths, n_states=4, n_iter=100,
              covariance_type="full", label="Model"):
    model = GaussianHMM(n_states=n_states, n_iter=n_iter, tol=1e-3,
                        random_state=42, covariance_type=covariance_type)
    t0 = time.time()
    flush_print(f"\n  Training {label} "
                f"({n_states} states, {n_iter} EM iters, cov={covariance_type})...")
    model.fit(X_concat, lengths)
    elapsed = time.time() - t0
    flush_print(f"\n  Transition Matrix ({n_states}x{n_states}):")
    for i, row in enumerate(model.transmat_):
        flush_print(f"    S{i}: [{', '.join(f'{p:.4f}' for p in row)}]")
    D = model.means_.shape[1]
    flush_print(f"  Emission Means (first PC): "
                f"{[f'{m:.4f}' for m in model.means_[:, 0]]}")
    flush_print(f"  Trained in {elapsed:.1f}s")
    return model


def train_both(X0, l0, X1, l1, n_states=4, n_iter=100, covariance_type="full"):
    flush_print("\n" + "=" * 72)
    flush_print("  PHASE 2: HMM TRAINING -- BAUM-WELCH (EM)")
    flush_print("=" * 72)
    m0 = train_hmm(X0, l0, n_states, n_iter, covariance_type, "NORMAL HMM")
    m1 = train_hmm(X1, l1, n_states, n_iter, covariance_type, "ARRHYTHMIA HMM")
    return m0, m1


# ============================================================================
#  PHASE 3 -- PREDICTION
# ============================================================================

def predict_all(m_norm, m_arrh, X_test, pca=None):
    """
    Score each test beat against both HMMs and pick the higher-likelihood class.

    If a fitted PCA object is supplied, each raw 187-feature beat is first
    projected to PCA space before scoring -- matching the training transform.
    """
    flush_print("\n" + "=" * 72)
    flush_print("  PHASE 3: VITERBI DECODING & PREDICTION")
    flush_print("=" * 72)

    n = X_test.shape[0]
    preds = np.zeros(n, dtype=int)
    lls = np.zeros((n, 2))
    paths = []

    t0 = time.time()
    every = max(1, n // 20)

    for i in range(n):
        # Apply the same PCA projection used during training
        if pca is not None:
            beat = pca.transform(X_test[i:i+1])  # (1, n_components)
        else:
            beat = X_test[i:i+1]                  # (1, 187) fallback

        ll0 = m_norm.score(beat)
        ll1 = m_arrh.score(beat)
        lls[i] = [ll0, ll1]

        if ll0 >= ll1:
            preds[i] = 0
            paths.append(m_norm.predict(beat))
        else:
            preds[i] = 1
            paths.append(m_arrh.predict(beat))

        if (i+1) % every == 0 or i == n-1:
            pct = 100*(i+1)/n
            el = time.time() - t0
            eta = el/(i+1)*(n-i-1)
            flush_print(f"\r  [{pct:5.1f}%] {i+1:>6,}/{n:,} | ETA {eta:.0f}s", end="")

    flush_print()
    total = time.time() - t0
    flush_print(f"\n  Done in {total:.1f}s ({n/total:.0f} beats/sec)")
    flush_print(f"  Predicted Normal: {(preds==0).sum():,} | Arrhythmia: {(preds==1).sum():,}")
    return preds, lls, paths


# ============================================================================
#  PHASE 4 -- Performance Dashboard & Visualization
# ============================================================================

def compute_metrics(y_true, y_pred):
    flush_print("\n" + "=" * 72)
    flush_print("  PHASE 4: PERFORMANCE METRICS & VISUALIZATION")
    flush_print("=" * 72)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    flush_print(f"\n  Classification Report:")
    for line in classification_report(y_true, y_pred,
                    target_names=["Normal","Arrhythmia"], digits=4, zero_division=0).strip().split("\n"):
        flush_print(f"    {line}")
    flush_print(f"\n  Confusion Matrix:")
    flush_print(f"                   Pred Normal   Pred Arrhythmia")
    flush_print(f"  True Normal      {cm[0,0]:>9,}       {cm[0,1]:>9,}")
    flush_print(f"  True Arrhythmia  {cm[1,0]:>9,}       {cm[1,1]:>9,}")
    return metrics, cm


def plot_dashboard(X_test, y_true, y_pred, metrics, cm,
                   m_norm, m_arrh, lls, n_show=40, return_fig=False):
    BG="#0d1117"; PAN="#161b22"; TXT="#e6edf3"; GRD="#30363d"
    GRN="#3fb950"; RED="#f85149"; BLU="#58a6ff"; GLD="#d29922"; PUR="#bc8cff"
    sns.set_theme(style="dark", rc={
        "figure.facecolor":BG,"axes.facecolor":PAN,"axes.edgecolor":GRD,
        "text.color":TXT,"axes.labelcolor":TXT,"xtick.color":TXT,
        "ytick.color":TXT,"grid.color":GRD,"grid.alpha":0.3})

    fig = plt.figure(figsize=(22,14), dpi=150)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2,3, height_ratios=[1.4,1], hspace=0.32, wspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    # Panel 1: ECG trace
    ax1 = fig.add_subplot(gs[0,:])
    nb = min(n_show, len(y_true))
    ecg = X_test[:nb].flatten()
    tax = np.arange(len(ecg))/125.0
    ax1.plot(tax, ecg, color=GRN, lw=0.6, alpha=0.95)
    for b in range(nb):
        if y_pred[b]==1:
            ax1.axvspan(b*187/125,(b+1)*187/125, color=RED, alpha=0.20)
    for b in range(nb):
        if y_true[b]==1:
            ax1.plot((b+0.5)*187/125, ax1.get_ylim()[1]*0.95, "v", color=RED, ms=5, alpha=0.7)
    ax1.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Voltage", fontsize=11, fontweight="bold")
    ax1.set_title("CONTINUOUS ECG TRACE -- ARRHYTHMIA DETECTION OVERLAY",
                  fontsize=14, fontweight="bold", color=BLU, pad=12)
    lh = [mpatches.Patch(fc=GRN, alpha=0.6, label="Normal Beat"),
          mpatches.Patch(fc=RED, alpha=0.35, label="Predicted Arrhythmia"),
          plt.Line2D([0],[0], marker="v", color="w", mfc=RED, ms=8,
                     label="True Arrhythmia", ls="None")]
    ax1.legend(handles=lh, loc="upper right", fontsize=9,
               facecolor=PAN, edgecolor=GRD, labelcolor=TXT)
    ax1.grid(True, lw=0.3)

    # Panel 2: Confusion matrix
    ax2 = fig.add_subplot(gs[1,0])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="RdYlGn_r",
                xticklabels=["Normal","Arrhythmia"],
                yticklabels=["Normal","Arrhythmia"],
                linewidths=1.5, linecolor=GRD,
                annot_kws={"size":13,"fontweight":"bold"},
                cbar_kws={"shrink":0.75}, ax=ax2)
    ax2.set_xlabel("Predicted", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Actual", fontsize=10, fontweight="bold")
    ax2.set_title("CONFUSION MATRIX", fontsize=12, fontweight="bold", color=BLU, pad=10)

    # Panel 3: Transition matrix
    ax3 = fig.add_subplot(gs[1,1])
    ns = m_norm.transmat_.shape[0]
    im = ax3.imshow(m_norm.transmat_, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    for i in range(ns):
        for j in range(ns):
            v = m_norm.transmat_[i,j]
            ax3.text(j,i,f"{v:.3f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white" if v>0.5 else "black")
    ax3.set_xticks(range(ns)); ax3.set_yticks(range(ns))
    ax3.set_xticklabels([f"S{i}" for i in range(ns)])
    ax3.set_yticklabels([f"S{i}" for i in range(ns)])
    ax3.set_xlabel("To State", fontsize=10, fontweight="bold")
    ax3.set_ylabel("From State", fontsize=10, fontweight="bold")
    ax3.set_title("TRANSITION MATRIX (Normal HMM)",
                  fontsize=11, fontweight="bold", color=BLU, pad=10)
    plt.colorbar(im, ax=ax3, shrink=0.75)

    # Panel 4: All classification metrics
    ax4 = fig.add_subplot(gs[1,2])
    names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    vals   = [metrics["Accuracy"], metrics["Precision"],
              metrics["Recall"],   metrics["F1-Score"]]
    colors = [BLU, GLD, GRN, PUR]
    bars = ax4.barh(names, vals, color=colors, edgecolor=GRD, lw=0.8, height=0.5)
    for bar, val in zip(bars, vals):
        ax4.text(min(bar.get_width() + 0.01, 1.13),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=11,
                 fontweight="bold", color=TXT)
    ax4.set_xlim(0, 1.18)
    ax4.set_xlabel("Score", fontsize=10, fontweight="bold")
    ax4.set_title("CLASSIFICATION PERFORMANCE",
                  fontsize=12, fontweight="bold", color=BLU, pad=10)
    ax4.grid(axis="x", lw=0.3, alpha=0.5)
    ax4.invert_yaxis()   # Accuracy on top, F1 on bottom

    fig.suptitle("PREDICTIVE ARRHYTHMIA DETECTION -- GAUSSIAN HMM DASHBOARD",
                 fontsize=18, fontweight="bold", color=TXT, y=0.98)
    p = os.path.join(OUT_DIR, "ecg_dashboard.png")
    fig.savefig(p, facecolor=BG, bbox_inches="tight")
    flush_print(f"\n  Dashboard saved -> {p}")
    if return_fig:
        return fig
    plt.close(fig)


def plot_ll_separation(lls, y_true, return_fig=False):
    BG, PAN, TXT, GRD = "#0d1117","#161b22","#e6edf3","#30363d"
    fig, ax = plt.subplots(figsize=(12,5), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(PAN)
    d = lls[:,0] - lls[:,1]
    ax.hist(d[y_true==0], bins=80, alpha=0.6, color="#3fb950",
            label="True Normal", density=True, edgecolor="none")
    ax.hist(d[y_true==1], bins=80, alpha=0.6, color="#f85149",
            label="True Arrhythmia", density=True, edgecolor="none")
    ax.axvline(0, color="#58a6ff", ls="--", lw=1.5, label="Decision boundary")
    ax.set_xlabel("Delta-LL (Normal - Arrhythmia)", fontsize=11,
                  fontweight="bold", color=TXT)
    ax.set_ylabel("Density", fontsize=11, fontweight="bold", color=TXT)
    ax.set_title("LOG-LIKELIHOOD SEPARATION -- HMM DECISION BOUNDARY",
                 fontsize=14, fontweight="bold", color="#58a6ff", pad=12)
    ax.legend(fontsize=10, facecolor=PAN, edgecolor=GRD, labelcolor=TXT)
    ax.tick_params(colors=TXT); ax.grid(True, color=GRD, alpha=0.3)
    for s in ax.spines.values(): s.set_color(GRD)
    p = os.path.join(OUT_DIR, "log_likelihood_separation.png")
    fig.savefig(p, facecolor=BG, bbox_inches="tight")
    flush_print(f"  LL separation plot saved -> {p}")
    if return_fig:
        return fig
    plt.close(fig)


# ============================================================================
#  MAIN
# ============================================================================

def run_pipeline(n_per_class=200, n_states=4, n_iter=100,
                 n_components=25, covariance_type="full",
                 return_figs=False):
    """
    Full arrhythmia detection pipeline.

    Parameters
    ----------
    n_per_class   : int   -- training beats sampled per class (default 200)
    n_states      : int   -- HMM hidden states (default 4)
    n_iter        : int   -- Baum-Welch EM iterations (default 100)
    n_components  : int   -- PCA components (default 25; ~95% variance)
    covariance_type : str -- 'full' (default) or 'diag'
    return_figs   : bool  -- return figure objects instead of saving & closing
    """
    flush_print("\n" + "=" * 72)
    flush_print("  PREDICTIVE ARRHYTHMIA DETECTION VIA HIDDEN MARKOV MODELS")
    flush_print("  MIT-BIH Arrhythmia Database | Custom GaussianHMM | numpy/scipy")
    flush_print(f"  Config: states={n_states} | iters={n_iter} | "
                f"PCA={n_components} | cov={covariance_type}")
    flush_print("=" * 72 + "\n")
    t0 = time.time()

    # Phase 1
    X_tr, y_tr_raw, X_te, y_te_raw = load_data(TRAIN_CSV, TEST_CSV)
    y_tr = binarize_labels(y_tr_raw)
    y_te = binarize_labels(y_te_raw)
    X0, l0, X1, l1, pca = prepare_hmm_data(
        X_tr, y_tr, n_per_class=n_per_class, n_components=n_components)

    # Phase 2
    m0, m1 = train_both(X0, l0, X1, l1,
                         n_states=n_states, n_iter=n_iter,
                         covariance_type=covariance_type)

    # Phase 3  (pass PCA so test beats are projected the same way)
    preds, lls, paths = predict_all(m0, m1, X_te, pca=pca)

    # Phase 4
    metrics, cm = compute_metrics(y_te, preds)
    fig_dash = plot_dashboard(X_te, y_te, preds, metrics, cm, m0, m1, lls, return_fig=return_figs)
    fig_sep = plot_ll_separation(lls, y_te, return_fig=return_figs)

    flush_print(f"\n{'=' * 72}")
    flush_print(f"  PIPELINE COMPLETE -- Total: {time.time()-t0:.1f}s")
    flush_print(f"  Output: {OUT_DIR}")
    flush_print(f"{'=' * 72}\n")

    return {
        "metrics": metrics, "cm": cm, "fig_dash": fig_dash, "fig_sep": fig_sep,
        "preds": preds, "lls": lls, "y_true": y_te, "X_test": X_te,
        "X_train": X_tr, "y_train_raw": y_tr_raw, "y_test_raw": y_te_raw,
        "n_train": X_tr.shape[0], "n_test": X_te.shape[0],
        "n_per_class": n_per_class, "n_states": n_states, "n_iter": n_iter,
        "n_components": n_components, "covariance_type": covariance_type,
        "pca": pca, "elapsed": time.time() - t0,
    }


def main():
    run_pipeline(
        n_per_class=200,
        n_states=4,
        n_iter=100,
        n_components=25,
        covariance_type="full",
    )


if __name__ == "__main__":
    main()
