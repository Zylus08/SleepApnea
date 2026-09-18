"""
Generate publication figures from completed loss ablation results.
Reads from results/loss_ablation/*.json and results/loss_ablation/ablation_summary.csv
Writes to figures/
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "results/loss_ablation"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load summary CSV ──────────────────────────────────────────────────────────
rows = []
with open(os.path.join(RESULTS_DIR, "ablation_summary.csv")) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

ablations    = [r["ablation"] for r in rows]
auroc        = [float(r["test_auroc"]) for r in rows]
auprc        = [float(r["test_auprc"]) for r in rows]
ssl_loss     = [float(r["ssl_final_loss"]) for r in rows]

LABELS = {
    "A0_VanillaNTXent":  "A0: NT-Xent",
    "A1_TemporalNTXent": "A1: Temporal NT-Xent",
    "A2_NTXent_TempReg": "A2: NT-Xent + TempReg",
    "A3_PhysioCLR":      "A3: PhysioCLR",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

x = np.arange(len(ablations))
width = 0.35

# ── Fig 3a: AUROC + AUPRC grouped bar ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

bars1 = ax.bar(x - width/2, auroc, width, label="AUROC", color="#4C72B0", alpha=0.9, zorder=3)
bars2 = ax.bar(x + width/2, auprc, width, label="AUPRC", color="#DD8452", alpha=0.9, zorder=3)

# Chance lines
ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8, label="AUROC chance", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[a] for a in ablations], color="white", fontsize=9)
ax.set_ylabel("Score", color="white", fontsize=10)
ax.set_ylim(0, 0.75)
ax.set_title("Fig 3a — Loss Ablation: Downstream Performance\n(5 SSL epochs, 8 DS epochs, seed=42, ds008108)",
             color="white", fontsize=10, pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.grid(axis="y", color="#333", zorder=1)
ax.legend(facecolor="#1C1E26", labelcolor="white", fontsize=9)

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", color="white", fontsize=7)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, "fig3a_loss_ablation_downstream.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")

# ── Fig 3b: SSL training curves ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

for i, abl in enumerate(ablations):
    jpath = os.path.join(RESULTS_DIR, f"{abl}_metrics.json")
    with open(jpath) as f:
        data = json.load(f)
    history = data.get("ssl_history", [])
    if not history:
        continue
    epochs = [h["epoch"] for h in history]
    losses = [h["total"] for h in history]
    ax.plot(epochs, losses, marker="o", color=COLORS[i], label=LABELS[abl], linewidth=2, markersize=5)

ax.set_xlabel("SSL Epoch", color="white", fontsize=10)
ax.set_ylabel("SSL Total Loss", color="white", fontsize=10)
ax.set_title("Fig 3b — SSL Training Loss by Ablation", color="white", fontsize=10, pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.grid(color="#333")
ax.legend(facecolor="#1C1E26", labelcolor="white", fontsize=9)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, "fig3b_ssl_training_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")

# ── Fig 3c: Scatter AUROC vs SSL Final Loss ───────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

for i, abl in enumerate(ablations):
    ax.scatter(ssl_loss[i], auroc[i], color=COLORS[i], s=120, zorder=3, label=LABELS[abl])
    ax.annotate(LABELS[abl], (ssl_loss[i], auroc[i]),
                textcoords="offset points", xytext=(6, 4),
                color=COLORS[i], fontsize=7)

ax.set_xlabel("SSL Final Loss", color="white", fontsize=10)
ax.set_ylabel("Downstream AUROC", color="white", fontsize=10)
ax.set_title("Fig 3c — SSL Loss vs Downstream AUROC", color="white", fontsize=10, pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.grid(color="#333", zorder=1)
ax.legend(facecolor="#1C1E26", labelcolor="white", fontsize=7)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, "fig3c_ssl_loss_vs_auroc.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")

print("All figures written to figures/")
