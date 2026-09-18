"""
experiments/e2e_latency_benchmark.py
======================================
Measures TRUE end-to-end latency:
  raw EEG (1, 20, 3000) → PyTorch STFT → ONNX INT8 → logit

Also measures STFT-only latency separately.
Reports:
  - STFT latency (ms)
  - ONNX INT8 inference latency (ms)
  - Total end-to-end latency (ms)
  - Real-Time Factor (RTF): total_time / 30s window

Answers Q9: "What is the true end-to-end latency?"
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:
    print("pip install onnxruntime"); sys.exit(1)

from model import STFTEncoder2D

INT8_PATH = r'E:\SleepApnea\SleepApneaSSL\sleep_apnea_int8.onnx'
FP32_PATH = r'E:\SleepApnea\SleepApneaSSL\sleep_apnea_fp32.onnx'
WARMUP    = 50
RUNS      = 500
WINDOW_S  = 30.0  # 30-second EEG epoch

def bench(fn, warmup=50, runs=500):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return float(np.mean(t)), float(np.percentile(t, 50)), float(np.percentile(t, 99))

def main():
    print("="*60)
    print("  END-TO-END LATENCY BENCHMARK")
    print("="*60)

    # Setup
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    encoder.eval()
    dummy_raw = torch.randn(1, 20, 3000, dtype=torch.float32)

    # Pre-compute spectrogram for ONNX-only benchmark
    with torch.no_grad():
        dummy_spec = encoder.stft(dummy_raw)
    dummy_spec_np = dummy_spec.numpy()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    results = {}

    # --- 1. STFT only ---
    print("\n[*] Benchmarking STFT (PyTorch, no_grad)...")
    def stft_fn():
        with torch.no_grad():
            encoder.stft(dummy_raw)
    stft_mean, stft_p50, stft_p99 = bench(stft_fn, WARMUP, RUNS)
    print(f"  Mean: {stft_mean:.3f} ms | P50: {stft_p50:.3f} ms | P99: {stft_p99:.3f} ms")
    results['stft_mean_ms']  = stft_mean
    results['stft_p50_ms']   = stft_p50
    results['stft_p99_ms']   = stft_p99

    # --- 2. ONNX INT8 only (spec input) ---
    for onnx_path, label in [(FP32_PATH, 'fp32'), (INT8_PATH, 'int8')]:
        if not os.path.exists(onnx_path):
            print(f"  SKIP {label}: {onnx_path} not found"); continue
        print(f"\n[*] Benchmarking ONNX {label.upper()} (spectrogram input)...")
        sess = ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])
        iname = sess.get_inputs()[0].name
        def onnx_fn(s=sess, n=iname, d=dummy_spec_np):
            s.run(None, {n: d})
        mn, p50, p99 = bench(onnx_fn, WARMUP, RUNS)
        print(f"  Mean: {mn:.3f} ms | P50: {p50:.3f} ms | P99: {p99:.3f} ms")
        results[f'onnx_{label}_mean_ms'] = mn
        results[f'onnx_{label}_p50_ms']  = p50
        results[f'onnx_{label}_p99_ms']  = p99

    # --- 3. End-to-end (STFT + INT8 ONNX) ---
    if os.path.exists(INT8_PATH):
        print(f"\n[*] Benchmarking END-TO-END (STFT + INT8 ONNX)...")
        sess_int8 = ort.InferenceSession(INT8_PATH, opts, providers=['CPUExecutionProvider'])
        iname = sess_int8.get_inputs()[0].name

        def e2e_fn():
            with torch.no_grad():
                spec = encoder.stft(dummy_raw).numpy()
            sess_int8.run(None, {iname: spec})

        e2e_mean, e2e_p50, e2e_p99 = bench(e2e_fn, WARMUP, RUNS)
        rtf = (e2e_mean / 1000.0) / WINDOW_S
        print(f"  Mean: {e2e_mean:.3f} ms | P50: {e2e_p50:.3f} ms | P99: {e2e_p99:.3f} ms")
        print(f"  Real-Time Factor: {rtf:.6f}  ({rtf*100:.4f}% of real time)")
        results['e2e_mean_ms'] = e2e_mean
        results['e2e_p50_ms']  = e2e_p50
        results['e2e_p99_ms']  = e2e_p99
        results['e2e_rtf']     = rtf

    # --- Summary ---
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  {'Component':<25}  {'Mean (ms)':>10}  {'P99 (ms)':>10}")
    print(f"  {'-'*50}")
    print(f"  {'STFT (PyTorch)':<25}  {stft_mean:>10.3f}  {stft_p99:>10.3f}")
    if 'onnx_fp32_mean_ms' in results:
        print(f"  {'ONNX FP32':<25}  {results['onnx_fp32_mean_ms']:>10.3f}  {results['onnx_fp32_p99_ms']:>10.3f}")
    if 'onnx_int8_mean_ms' in results:
        print(f"  {'ONNX INT8':<25}  {results['onnx_int8_mean_ms']:>10.3f}  {results['onnx_int8_p99_ms']:>10.3f}")
    if 'e2e_mean_ms' in results:
        print(f"  {'END-TO-END (STFT+INT8)':<25}  {results['e2e_mean_ms']:>10.3f}  {results['e2e_p99_ms']:>10.3f}")
        print(f"\n  RTF (end-to-end): {results['e2e_rtf']:.6f}")
        print(f"  STFT share of e2e: {100*stft_mean/results['e2e_mean_ms']:.1f}%")
        results['stft_fraction_of_e2e'] = stft_mean / results['e2e_mean_ms']

    # Save
    import json
    out = r'E:\SleepApnea\SleepApneaSSL\results\e2e_latency_benchmark.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")

if __name__ == '__main__':
    main()
