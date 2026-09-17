"""
experiments/generate_figures.py
Generate all publication-quality figures from saved results.
Run AFTER all experiments have completed.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

FIG_DIR = r'E:\SleepApnea\SleepApneaSSL\figures'
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    'A0': '#4C72B0',
    'A1': '#DD8452',
    'A2': '#55A868',
    'A3': '#C44E52',
}

# ── Figure 3: Loss Ablation ───────────────────────────────────────────────────
def fig3_loss_ablation():
    abl_dir = r'E:\SleepApnea\SleepApneaSSL\results\loss_ablation'
    summary = os.path.join(abl_dir, 'ablation_summary.csv')
    if not os.path.exists(summary):
        print("[skip] Fig3: ablation_summary.csv not found"); return

    import csv
    rows = []
    with open(summary) as f:
        rows = list(csv.DictReader(f))

    names  = [r['ablation'] for r in rows]
    aurocs = [float(r['test_auroc']) for r in rows]
    auprcs = [float(r['test_auprc']) for r in rows]
    ssl_losses = [float(r['ssl_final_loss']) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = [PALETTE.get(n.split('_')[0], '#888') for n in names]
    short = [n.split('_', 1)[1] if '_' in n else n for n in names]

    for ax, vals, title in zip(axes,
                               [aurocs, auprcs, ssl_losses],
                               ['Test AUROC', 'Test AUPRC', 'SSL Final Loss']):
        bars = ax.bar(short, vals, color=colors, edgecolor='white', width=0.6)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Figure 3 — Loss Ablation: SSL Objective Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_loss_ablation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved {path}")

# ── Figure 4/5: Lambda ablation ───────────────────────────────────────────────
def fig45_lambda_ablation():
    lam_dir = r'E:\SleepApnea\SleepApneaSSL\results\lambda_ablation'
    if not os.path.exists(lam_dir):
        print("[skip] Fig4/5: lambda ablation results not found"); return

    import csv
    for fname, title, figname in [
        ('lambda_decay_results.csv',    'λ_decay (TemporalNTXent)', 'fig4_lambda_decay.png'),
        ('lambda_temporal_results.csv', 'λ_temporal (PhysioCLR)',   'fig5_lambda_temporal.png'),
    ]:
        fpath = os.path.join(lam_dir, fname)
        if not os.path.exists(fpath): continue
        rows = list(csv.DictReader(open(fpath)))
        lams   = [float(r['lambda']) for r in rows]
        aurocs = [float(r['test_auroc']) for r in rows]
        auprcs = [float(r['test_auprc']) for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(lams, aurocs, 'o-', color='#4C72B0', lw=2)
        ax1.set_xscale('log'); ax1.set_xlabel(title); ax1.set_ylabel('Test AUROC')
        ax1.set_title(f'{title} vs AUROC'); ax1.grid(alpha=0.3)
        ax2.plot(lams, auprcs, 's-', color='#DD8452', lw=2)
        ax2.set_xscale('log'); ax2.set_xlabel(title); ax2.set_ylabel('Test AUPRC')
        ax2.set_title(f'{title} vs AUPRC'); ax2.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, figname)
        plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"[+] Saved {path}")

# ── Figure 6: Few-shot curve ──────────────────────────────────────────────────
def fig6_few_shot():
    csv_path = r'E:\SleepApnea\SleepApneaSSL\transfer_results.csv'
    if not os.path.exists(csv_path):
        print("[skip] Fig6: transfer_results.csv not found"); return
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    fracs  = [float(r['Fraction'])*100 for r in rows]
    w_aucs = [float(r['Window_AUC']) for r in rows]
    p_aucs = [float(r['Patient_AUC']) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, p_aucs, 'o-', lw=2, color='#4C72B0', label='Patient-Level AUC')
    ax.plot(fracs, w_aucs, 's--', lw=2, color='#DD8452', label='Window-Level AUC')
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 100]); ax.set_xticklabels(['1%', '5%', '10%', '100%'])
    ax.set_xlabel('Training Data Fraction'); ax.set_ylabel('AUC')
    ax.set_title('Figure 6 — Few-Shot Transfer: Sample Efficiency\n'
                 '(NOTE: checkpoint selection used test AUC — results are optimistic)',
                 fontsize=11)
    ax.legend(); ax.grid(alpha=0.3)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.7, label='Random baseline')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_few_shot_curve.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[+] Saved {path}")

# ── Figure 11: Quantization tradeoff ─────────────────────────────────────────
def fig11_quantization():
    q_path = r'E:\SleepApnea\SleepApneaSSL\results\quantization_results.json'
    if not os.path.exists(q_path):
        print("[skip] Fig11: quantization_results.json not found"); return
    with open(q_path) as f:
        data = json.load(f)

    models = list(data.keys())
    sizes   = [data[m].get('size_mb', 0) for m in models]
    latency = [data[m].get('mean_latency_ms', 0) for m in models]
    aurocs  = [data[m].get('auroc', 0) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52'][:len(models)]
    for ax, vals, ylabel in zip(axes, [sizes, latency, aurocs],
                                ['Model Size (MB)', 'Mean Latency (ms)', 'AUROC']):
        ax.bar(models, vals, color=colors, edgecolor='white', width=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v*1.01, f'{v:.3f}' if ylabel=='AUROC' else f'{v:.1f}', ha='center', fontsize=9)
        ax.set_title(ylabel); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Figure 11 — FP32 vs INT8 Deployment Tradeoff', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig11_quantization.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[+] Saved {path}")

# ── Figure 1: Pipeline diagram ────────────────────────────────────────────────
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis('off')

    steps = [
        ('Raw EEG\n(ds008108)', 0.5),
        ('Anti-alias\nResample\n100 Hz', 2.3),
        ('STFT\nSpectrogram', 4.1),
        ('SpectralSubband\nMasking', 5.9),
        ('STFTEncoder2D\n(CNN 2D)', 7.7),
        ('PhysioCLR\nInfoNCE +\nTemporal', 9.5),
        ('2-layer MLP\nHead', 11.3),
        ('OSA\nPrediction', 13.1),
        ('ONNX\nINT8 Deploy', 14.9),
    ]
    for label, x in steps:
        rect = mpatches.FancyBboxPatch((x-0.8, 0.8), 1.6, 2.4, boxstyle='round,pad=0.1',
                                        facecolor='#E8F4FD', edgecolor='#4C72B0', lw=1.5)
        ax.add_patch(rect)
        ax.text(x, 2.0, label, ha='center', va='center', fontsize=7.5, fontweight='bold')

    for i in range(len(steps)-1):
        x0 = steps[i][1] + 0.8
        x1 = steps[i+1][1] - 0.8
        ax.annotate('', xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=1.5))

    ax.set_title('Figure 1 — SleepApnea SSL Research Pipeline', fontsize=13, fontweight='bold', pad=10)
    path = os.path.join(FIG_DIR, 'fig1_pipeline.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[+] Saved {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig1_pipeline()
    fig3_loss_ablation()
    fig45_lambda_ablation()
    fig6_few_shot()
    fig11_quantization()
    print("\n[+] Figure generation complete. Check figures/ directory.")
