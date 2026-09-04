"""
Centered Kernel Alignment (CKA) & MMD — Domain Shift Analysis
===============================================================
Quantifies representation similarity between source (ds008108) and
target (Sleep-EDF) cohorts at each encoder layer to justify domain-shift
robustness claims in the experiments section.

Metrics
-------
- **Linear CKA** (Kornblith et al., 2019): Measures representational
  similarity between two sets of activations. CKA ∈ [0, 1] where 1.0
  means identical representation geometry.
  
      CKA(K, L) = ‖L^T K‖_F² / (‖K^T K‖_F · ‖L^T L‖_F)

- **MMD** (Gretton et al., 2012): Maximum Mean Discrepancy with RBF
  kernel. Measures distributional distance. Lower = more similar.

Usage
-----
    python cka_analysis.py

    # Custom paths
    python cka_analysis.py \\
        --source-dir E:\\SleepApneaProcessed \\
        --target-dir E:\\sleep-edf-database-expanded-1.0.0\\sleep-cassette \\
        --n-samples 500

Output
------
    - Console: CKA & MMD values per layer
    - cka_heatmap.png: Layer-wise CKA similarity heatmap
    - cka_results.npz: Raw CKA/MMD values for downstream analysis
"""

import argparse
import gc
import glob
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Project imports
from model import STFTEncoder2D

# Suppress MNE warnings
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 1. CKA IMPLEMENTATION
# ==========================================
def linear_HSIC(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the linear Hilbert-Schmidt Independence Criterion (HSIC).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    Y : np.ndarray, shape (n, q)

    Returns
    -------
    hsic : float
    """
    n = X.shape[0]
    # Center the Gram matrices
    H = np.eye(n) - np.ones((n, n)) / n
    K = X @ X.T  # Linear kernel
    L = Y @ Y.T
    return float(np.trace(K @ H @ L @ H)) / ((n - 1) ** 2)


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear Centered Kernel Alignment between two representation
    matrices.

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) · HSIC(Y, Y))

    Parameters
    ----------
    X : np.ndarray, shape (n, p) — activations from source cohort
    Y : np.ndarray, shape (n, q) — activations from target cohort

    Returns
    -------
    cka : float ∈ [0, 1]
    """
    hsic_xy = linear_HSIC(X, Y)
    hsic_xx = linear_HSIC(X, X)
    hsic_yy = linear_HSIC(Y, Y)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


# ==========================================
# 2. MMD IMPLEMENTATION
# ==========================================
def compute_MMD(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.

    MMD²(X, Y) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]

    Parameters
    ----------
    X : np.ndarray, shape (n, d) — source representations
    Y : np.ndarray, shape (m, d) — target representations
    gamma : float or None — RBF bandwidth (1 / (2σ²)). If None,
            uses the median heuristic.

    Returns
    -------
    mmd : float — MMD distance (square root of MMD²)
    """
    if gamma is None:
        # Median heuristic for bandwidth selection
        XY = np.vstack([X, Y])
        from scipy.spatial.distance import pdist
        dists = pdist(XY, metric="sqeuclidean")
        median_dist = np.median(dists)
        gamma = 1.0 / (median_dist + 1e-8)

    def rbf_kernel(A, B, gamma):
        # ||a - b||² = ||a||² + ||b||² - 2·a·b
        sq_A = np.sum(A ** 2, axis=1, keepdims=True)
        sq_B = np.sum(B ** 2, axis=1, keepdims=True)
        D = sq_A + sq_B.T - 2.0 * A @ B.T
        return np.exp(-gamma * D)

    K_xx = rbf_kernel(X, X, gamma)
    K_yy = rbf_kernel(Y, Y, gamma)
    K_xy = rbf_kernel(X, Y, gamma)

    n = X.shape[0]
    m = Y.shape[0]

    # Unbiased MMD² estimator
    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)

    mmd_sq = (K_xx.sum() / (n * (n - 1))
              + K_yy.sum() / (m * (m - 1))
              - 2.0 * K_xy.sum() / (n * m))

    return float(np.sqrt(max(mmd_sq, 0.0)))


# ==========================================
# 3. LAYER-WISE FEATURE EXTRACTION
# ==========================================
class LayerHookExtractor:
    """
    Registers forward hooks on specific layers of the STFTEncoder2D
    to capture intermediate activations.

    Captures activations at three stages:
        - post_conv1 : After first conv block + BN + GELU + MaxPool
        - post_conv2 : After second conv block
        - post_fc    : After the final FC layer (embedding)
    """

    def __init__(self, encoder: STFTEncoder2D):
        self.encoder = encoder
        self.activations = {}
        self._hooks = []

        # Register hooks on the three conv blocks and the FC layer
        # STFTEncoder2D.conv_blocks is a nn.Sequential with 3 groups of 4 layers each:
        #   [0-3]: Conv2d(20,32) -> BN -> GELU -> MaxPool
        #   [4-7]: Conv2d(32,64) -> BN -> GELU -> MaxPool
        #   [8-11]: Conv2d(64,128) -> BN -> GELU -> AdaptiveAvgPool
        self._hooks.append(
            encoder.conv_blocks[3].register_forward_hook(
                self._make_hook("post_conv1")
            )
        )
        self._hooks.append(
            encoder.conv_blocks[7].register_forward_hook(
                self._make_hook("post_conv2")
            )
        )
        self._hooks.append(
            encoder.fc.register_forward_hook(
                self._make_hook("post_fc")
            )
        )

    def _make_hook(self, name):
        def hook(module, input, output):
            # Flatten spatial dimensions, keep batch dim
            if output.ndim > 2:
                self.activations[name] = output.flatten(1).detach().cpu().numpy()
            else:
                self.activations[name] = output.detach().cpu().numpy()
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def extract_source_features(
    encoder: STFTEncoder2D,
    data_dir: str,
    n_samples: int,
    device: torch.device,
    hook_extractor: LayerHookExtractor,
) -> dict:
    """
    Extract layer-wise features from source cohort (ds008108 preprocessed .pt files).

    Returns
    -------
    features : dict[str, np.ndarray] — layer_name -> (n_samples, dim)
    """
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    all_features = {name: [] for name in ["post_conv1", "post_conv2", "post_fc"]}
    collected = 0

    encoder.eval()
    with torch.no_grad():
        for f in tqdm(pt_files, desc="Source features"):
            if collected >= n_samples:
                break

            data = torch.load(f, map_location="cpu", weights_only=True)

            # Handle various tensor formats from the dataset
            if isinstance(data, dict):
                signals = data.get("x", data.get("signals", None))
            elif isinstance(data, (tuple, list)):
                signals = data[0]
            else:
                signals = data

            if signals is None:
                continue

            # Ensure 3D: (N_windows, 20, 3000)
            if signals.dim() == 2:
                if signals.size(0) == 20:
                    n_win = signals.size(1) // 3000
                    signals = signals[:, : n_win * 3000].view(20, n_win, 3000).permute(1, 0, 2)
                else:
                    n_win = signals.size(0) // 3000
                    signals = signals[: n_win * 3000, :].view(n_win, 3000, 20).permute(0, 2, 1)

            # Sample a subset of windows from this subject
            n_win = signals.size(0)
            remaining = n_samples - collected
            use_n = min(n_win, remaining, 32)  # Cap per-subject contribution
            indices = torch.randperm(n_win)[:use_n]
            batch = signals[indices].float().to(device)

            # Forward pass triggers hooks
            _ = encoder(batch)

            for name in all_features:
                all_features[name].append(hook_extractor.activations[name])

            collected += use_n
            del data, signals, batch
            gc.collect()

    return {name: np.concatenate(arrs)[:n_samples] for name, arrs in all_features.items()}


def extract_target_features(
    encoder: STFTEncoder2D,
    data_dir: str,
    n_samples: int,
    device: torch.device,
    hook_extractor: LayerHookExtractor,
) -> dict:
    """
    Extract layer-wise features from target cohort (Sleep-EDF .edf files).

    Uses the same extraction logic as cross_cohort_eval.py for consistency.
    """
    from cross_cohort_eval import match_sleep_edf_files, extract_epochs

    file_pairs = match_sleep_edf_files(data_dir)

    all_features = {name: [] for name in ["post_conv1", "post_conv2", "post_fc"]}
    collected = 0

    encoder.eval()
    with torch.no_grad():
        for psg, hyp in tqdm(file_pairs, desc="Target features"):
            if collected >= n_samples:
                break

            X, y, _ = extract_epochs(psg, hyp)
            if X is None or len(X) == 0:
                continue

            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Match to 20-channel input
            if X_tensor.ndim == 3 and X_tensor.shape[1] == 1:
                X_tensor = X_tensor.repeat(1, 20, 1)
            elif X_tensor.ndim == 2:
                X_tensor = X_tensor.unsqueeze(1).repeat(1, 20, 1)

            remaining = n_samples - collected
            use_n = min(X_tensor.size(0), remaining, 32)
            indices = torch.randperm(X_tensor.size(0))[:use_n]
            batch = X_tensor[indices].to(device)

            _ = encoder(batch)

            for name in all_features:
                all_features[name].append(hook_extractor.activations[name])

            collected += use_n
            del X, X_tensor, batch
            gc.collect()

    return {name: np.concatenate(arrs)[:n_samples] for name, arrs in all_features.items()}


# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_cka_heatmap(cka_values: dict, mmd_values: dict, output_path: str):
    """
    Generate a publication-quality CKA heatmap and MMD bar chart.
    """
    layers = list(cka_values.keys())
    cka_arr = np.array([cka_values[l] for l in layers]).reshape(1, -1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

    # Panel A: CKA heatmap
    ax = axes[0]
    im = ax.imshow(cka_arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace("_", "\n") for l in layers], fontsize=10)
    ax.set_yticks([0])
    ax.set_yticklabels(["Source → Target"], fontsize=10)
    ax.set_title("(A) Linear CKA: Representational Similarity", fontsize=12, fontweight="bold")

    # Annotate cells
    for j, layer in enumerate(layers):
        ax.text(j, 0, f"{cka_values[layer]:.3f}", ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if cka_values[layer] > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.6, label="CKA Similarity")

    # Panel B: MMD bar chart
    ax = axes[1]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(range(len(layers)), [mmd_values[l] for l in layers], color=colors, alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace("_", "\n") for l in layers], fontsize=10)
    ax.set_ylabel("MMD Distance", fontsize=11)
    ax.set_title("(B) MMD: Distribution Distance", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate bars
    for bar, layer in zip(bars, layers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mmd_values[layer]:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] CKA heatmap saved to {output_path}")


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="CKA & MMD domain shift analysis between source and target cohorts."
    )
    parser.add_argument(
        "--source-dir", type=str,
        default=r"E:\SleepApneaProcessed",
        help="Directory containing source cohort preprocessed .pt files."
    )
    parser.add_argument(
        "--target-dir", type=str,
        default=r"E:\sleep-edf-database-expanded-1.0.0\sleep-cassette",
        help="Directory containing target cohort Sleep-EDF .edf files."
    )
    parser.add_argument(
        "--model-path", type=str,
        default=r"E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth",
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of windows to sample from each cohort (default: 500)."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=r"E:\SleepApnea\SleepApneaSSL",
        help="Directory for output files."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # -----------------------------------------------------------------------
    # 1. Load encoder (extract encoder weights from SleepApneaClassifier)
    # -----------------------------------------------------------------------
    print("[*] Loading encoder weights...")
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128).to(device)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    # Filter to encoder weights only (strip 'encoder.' prefix from classifier checkpoint)
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    if encoder_state:
        encoder.load_state_dict(encoder_state, strict=True)
        print("[+] Loaded encoder weights from classifier checkpoint.")
    else:
        # Assume the checkpoint IS the encoder directly
        encoder.load_state_dict(state_dict, strict=False)
        print("[+] Loaded encoder weights directly.")

    encoder.eval()

    # -----------------------------------------------------------------------
    # 2. Register layer hooks
    # -----------------------------------------------------------------------
    hook_extractor = LayerHookExtractor(encoder)

    # -----------------------------------------------------------------------
    # 3. Extract features from both cohorts
    # -----------------------------------------------------------------------
    print(f"\n[*] Extracting {args.n_samples} windows from SOURCE cohort...")
    source_features = extract_source_features(
        encoder, args.source_dir, args.n_samples, device, hook_extractor
    )

    print(f"\n[*] Extracting {args.n_samples} windows from TARGET cohort...")
    target_features = extract_target_features(
        encoder, args.target_dir, args.n_samples, device, hook_extractor
    )

    hook_extractor.remove_hooks()

    # -----------------------------------------------------------------------
    # 4. Compute CKA and MMD at each layer
    # -----------------------------------------------------------------------
    layers = ["post_conv1", "post_conv2", "post_fc"]

    print("\n" + "=" * 60)
    print("  REPRESENTATION GEOMETRY ANALYSIS")
    print("  Source: ds008108 (OSA cohort) | Target: Sleep-EDF")
    print("=" * 60)

    cka_values = {}
    mmd_values = {}

    for layer in layers:
        X_src = source_features[layer]
        X_tgt = target_features[layer]

        # Use the minimum sample count between cohorts
        n = min(X_src.shape[0], X_tgt.shape[0])
        X_src = X_src[:n]
        X_tgt = X_tgt[:n]

        cka = linear_CKA(X_src, X_tgt)
        mmd = compute_MMD(X_src, X_tgt)

        cka_values[layer] = cka
        mmd_values[layer] = mmd

        print(f"  {layer:15s}  |  CKA: {cka:.4f}  |  MMD: {mmd:.4f}  "
              f"|  dim: {X_src.shape[1]}")

    # -----------------------------------------------------------------------
    # 5. Save results and generate visualization
    # -----------------------------------------------------------------------
    npz_path = os.path.join(args.output_dir, "cka_results.npz")
    np.savez(
        npz_path,
        cka_values=np.array([cka_values[l] for l in layers]),
        mmd_values=np.array([mmd_values[l] for l in layers]),
        layer_names=np.array(layers),
    )
    print(f"\n[+] CKA/MMD values saved to {npz_path}")

    plot_path = os.path.join(args.output_dir, "cka_heatmap.png")
    plot_cka_heatmap(cka_values, mmd_values, plot_path)

    # -----------------------------------------------------------------------
    # 6. Interpretation summary
    # -----------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  INTERPRETATION")
    print("-" * 60)
    fc_cka = cka_values["post_fc"]
    conv1_cka = cka_values["post_conv1"]

    if fc_cka > 0.7:
        print("  ✓ High CKA at post_fc suggests strong representational alignment.")
        print("    The encoder learns domain-invariant features that transfer well.")
    elif fc_cka > 0.4:
        print("  ~ Moderate CKA at post_fc. Some domain shift is present but")
        print("    representations retain partial structural similarity.")
    else:
        print("  ✗ Low CKA at post_fc indicates significant representation divergence.")
        print("    Consider domain adaptation or layer-wise fine-tuning.")

    if conv1_cka > fc_cka:
        print("  → CKA decreases through the network: early features are more")
        print("    universal while later features become task-specific.")
    else:
        print("  → CKA increases through the network: deeper representations")
        print("    become more aligned, suggesting effective invariant learning.")


if __name__ == "__main__":
    main()
