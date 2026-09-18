"""
experiments/int8_accuracy_comparison.py
=========================================
Compares FP32 vs INT8 ONNX model accuracy on the ablation test set.
Measures: logit correlation, sigmoid prob correlation, patient-level
AUROC/AUPRC for both, and mean absolute logit difference.

Answers Q10: "Does INT8 materially alter predictions?"
"""
import os, sys, glob, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime"); sys.exit(1)

from downstream_finetune import load_bids_labels, LabeledStreamingDataset

SEED      = 42
DATA_DIR  = r'E:\SleepApneaProcessed'
TSV_PATH  = r'E:\SleepApnea\participants.tsv'
FP32_PATH = r'E:\SleepApnea\SleepApneaSSL\sleep_apnea_fp32.onnx'
INT8_PATH = r'E:\SleepApnea\SleepApneaSSL\sleep_apnea_int8.onnx'

random.seed(SEED); np.random.seed(SEED)

def build_session(path):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])

def run_onnx_inference(session, loader, device='cpu'):
    """Run ONNX session on a DataLoader that yields (windows, labels, pids).
    Windows are (B, 20, 3000) raw EEG — we STFT them in PyTorch first
    to match the ONNX model's expected spectrogram input.
    """
    # Import encoder just to run STFT
    from model import STFTEncoder2D
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    encoder.eval()

    input_name = session.get_inputs()[0].name
    pat_logits  = {}
    pat_targets = {}

    for x, y, pids in loader:
        # Compute STFT spectrogram (same path as training)
        with torch.no_grad():
            specs = encoder.stft(x)  # (B, 20, F, T)
        specs_np = specs.numpy()
        logits = session.run(None, {input_name: specs_np})[0]  # (B,)

        for logit, lbl, pid in zip(logits, y.numpy(), pids.numpy()):
            pid = int(pid)
            pat_logits.setdefault(pid, []).append(float(logit))
            pat_targets[pid] = float(lbl)

    ids   = list(pat_logits)
    pl    = np.array([np.mean(pat_logits[p])   for p in ids])  # mean patient logit
    pp    = 1.0 / (1.0 + np.exp(-pl))                          # sigmoid
    pt    = np.array([pat_targets[p] for p in ids])
    return pt, pl, pp

def main():
    for path, name in [(FP32_PATH, 'FP32'), (INT8_PATH, 'INT8')]:
        if not os.path.exists(path):
            print(f"MISSING: {path}"); sys.exit(1)

    label_map = load_bids_labels(TSV_PATH)
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))
    pids, pid_to_file = [], {}
    for f in all_files:
        digits = re.findall(r'\d+', os.path.basename(f))
        if digits:
            pid = int(digits[0])
            if pid in label_map:
                pids.append(pid); pid_to_file[pid] = f
    pids = list(set(pids))
    if len(pids) < 10:
        print("SKIP: insufficient data"); sys.exit(1)

    labels = [label_map[p] for p in pids]
    trainval_ids, test_ids = train_test_split(pids, test_size=0.20, stratify=labels, random_state=SEED)
    tv_labels = [label_map[p] for p in trainval_ids]
    _, _ = train_test_split(trainval_ids, test_size=0.25, stratify=tv_labels, random_state=SEED)
    test_files = [pid_to_file[p] for p in test_ids]

    loader = DataLoader(
        LabeledStreamingDataset(test_files, label_map, is_train=False),
        batch_size=64, shuffle=False
    )

    print("="*60)
    print("  INT8 vs FP32 ACCURACY COMPARISON")
    print("="*60)
    print(f"  Test subjects: {len(test_ids)} | Files: {len(test_files)}")
    print("\n[*] Running FP32 inference...")
    sess_fp32 = build_session(FP32_PATH)
    pt, fp32_logits, fp32_probs = run_onnx_inference(sess_fp32, loader)

    print("[*] Running INT8 inference...")
    sess_int8 = build_session(INT8_PATH)
    _, int8_logits, int8_probs = run_onnx_inference(sess_int8, loader)

    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)

    # --- Prediction agreement ---
    pr, _ = pearsonr(fp32_logits, int8_logits)
    sr, _ = spearmanr(fp32_logits, int8_logits)
    mae   = float(np.mean(np.abs(fp32_logits - int8_logits)))
    max_ae = float(np.max(np.abs(fp32_logits - int8_logits)))
    prob_mae = float(np.mean(np.abs(fp32_probs - int8_probs)))

    print(f"\n--- Logit Agreement (patient-level) ---")
    print(f"  Pearson r:          {pr:.6f}")
    print(f"  Spearman r:         {sr:.6f}")
    print(f"  Mean |Δlogit|:      {mae:.6f}")
    print(f"  Max  |Δlogit|:      {max_ae:.6f}")
    print(f"  Mean |Δprob|:       {prob_mae:.6f}")

    # --- Per-model AUROC/AUPRC ---
    if len(set(pt)) >= 2:
        fp32_auroc = roc_auc_score(pt, fp32_probs)
        int8_auroc = roc_auc_score(pt, int8_probs)
        fp32_auprc = average_precision_score(pt, fp32_probs)
        int8_auprc = average_precision_score(pt, int8_probs)
        auroc_delta = abs(fp32_auroc - int8_auroc)
        auprc_delta = abs(fp32_auprc - int8_auprc)

        print(f"\n--- Downstream Metrics (N={len(pt)} patients) ---")
        print(f"  {'Metric':<12}  {'FP32':>8}  {'INT8':>8}  {'|Δ|':>8}")
        print(f"  {'-'*44}")
        print(f"  {'AUROC':<12}  {fp32_auroc:>8.4f}  {int8_auroc:>8.4f}  {auroc_delta:>8.4f}")
        print(f"  {'AUPRC':<12}  {fp32_auprc:>8.4f}  {int8_auprc:>8.4f}  {auprc_delta:>8.4f}")

        # Rank agreement: do they order patients the same?
        fp32_rank = np.argsort(np.argsort(-fp32_logits))
        int8_rank = np.argsort(np.argsort(-int8_logits))
        rank_mae  = float(np.mean(np.abs(fp32_rank.astype(float) - int8_rank.astype(float))))
        print(f"\n  Rank agreement (mean |Δrank|): {rank_mae:.3f} (0=identical, {len(pt)/2:.0f}=worst)")
    else:
        print("  SKIP: single class in test set"); fp32_auroc = int8_auroc = float('nan')
        fp32_auprc = int8_auprc = float('nan'); auroc_delta = auprc_delta = float('nan')

    # --- File sizes ---
    fp32_mb = os.path.getsize(FP32_PATH) / 1e6
    int8_mb = os.path.getsize(INT8_PATH) / 1e6
    print(f"\n--- Model Sizes ---")
    print(f"  FP32: {fp32_mb:.2f} MB")
    print(f"  INT8: {int8_mb:.2f} MB  ({100*(1-int8_mb/fp32_mb):.1f}% smaller)")

    # --- Verdict ---
    print(f"\n--- Verdict (Q10: Does INT8 materially alter predictions?) ---")
    if mae < 0.05 and prob_mae < 0.02:
        verdict = "NO — logit difference is negligible (<0.05 mean absolute)"
    elif mae < 0.2:
        verdict = "MARGINAL — small but measurable logit shift"
    else:
        verdict = "YES — substantial logit difference"
    print(f"  {verdict}")

    # Save results
    import json
    results = {
        'n_test_patients': int(len(pt)),
        'pearson_r': float(pr),
        'spearman_r': float(sr),
        'mean_abs_logit_diff': mae,
        'max_abs_logit_diff': max_ae,
        'mean_abs_prob_diff': prob_mae,
        'fp32_auroc': float(fp32_auroc),
        'int8_auroc': float(int8_auroc),
        'auroc_delta': float(auroc_delta),
        'fp32_auprc': float(fp32_auprc),
        'int8_auprc': float(int8_auprc),
        'auprc_delta': float(auprc_delta),
        'fp32_size_mb': fp32_mb,
        'int8_size_mb': int8_mb,
        'size_reduction_pct': float(100*(1-int8_mb/fp32_mb)),
        'verdict': verdict,
    }
    out = r'E:\SleepApnea\SleepApneaSSL\results\int8_accuracy_comparison.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")

if __name__ == '__main__':
    main()
