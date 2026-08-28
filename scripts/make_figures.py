"""Generate standalone vector PDF figures (flowcharts + charts) for IndoSmish.
Each figure is its own PDF in figures/.  Run: python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)

# ---- palette (validated) ----
NAVY = "#1b356b"; BLUE = "#3366ff"; BLUE_LT = "#e8f0ff"; ORANGE = "#e8833a"
INK = "#222a38"; MUTED = "#5d646e"; LIGHT = "#f3f6fc"; LINE = "#cdd6ea"
GREEN = "#1fa86b"; AMBER = "#d99300"; RED = "#d83a4a"; GREY = "#9aa3b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9d1e0",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#e9edf5",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); print("wrote", name)


def rbox(ax, x, y, w, h, text, fc, ec, tc, fs=10, bold=True, sub=None, subc=MUTED):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    if sub:
        ax.text(x + w/2, y + h*0.63, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=fs-2.5, color=subc)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=BLUE, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=color, linestyle=ls, shrinkA=2, shrinkB=2))


def blank_ax(w=13, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    ax.grid(False)
    return fig, ax


# ===== FIGURE 1 — System pipeline flowchart =====
fig, ax = blank_ax(13, 5.6)
steps = [("Incoming SMS", "native / Roman /\ncode-mixed"), ("Script &\nLanguage ID", "token-level tags"),
         ("Normalization", "raw + optional\nIndicXlit"), ("IndicBERT\n(ONNX)", "on-device @ fp32"),
         ("Decision", "ham / spam /\nsmishing"), ("User action", "allow / warn /\nquarantine")]
n = len(steps); w = 1.9; gap = 0.18; x0 = (13 - (n*w + (n-1)*gap)) / 2
cy = 3.4
for i, (t, sub) in enumerate(steps):
    x = x0 + i*(w+gap)
    detector = (i == 3)
    rbox(ax, x, cy, w, 1.2, t, NAVY if detector else LIGHT, NAVY if detector else LINE,
         "white" if detector else INK, fs=10.5, sub=sub, subc="#c7d2ec" if detector else MUTED)
    if i < n-1:
        arrow(ax, x+w, cy+0.6, x+w+gap, cy+0.6)
# support branches
for i, (bx, t, sub) in enumerate([(1.4, "Governed corpus", "dedup group/time splits"),
                                   (5.35, "Latency / memory / energy", "robustness harness"),
                                   (9.3, "Federated update", "privacy + poisoning ctrl")]):
    rbox(ax, bx, 1.5, 2.6, 0.9, t, BLUE_LT, BLUE, BLUE, fs=9.5, sub=sub)
arrow(ax, 2.7, 2.4, 2.7, cy, color=GREY, ls=(0,(3,3)))
arrow(ax, 6.65, 2.4, 6.65, cy, color=GREY, ls=(0,(3,3)))
arrow(ax, cy+7.2, 2.4, 10.6, cy, color=GREY, ls=(0,(3,3)))
ax.add_patch(FancyBboxPatch((0.4, 0.3), 12.2, 0.75, boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=1.0, edgecolor=LINE, facecolor=LIGHT))
ax.text(6.5, 0.67, "Local classification path — any networked lookup is disclosed and evaluated separately.",
        ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
ax.set_title("Figure 1.  On-device smishing detection pipeline (dependency-aware architecture)",
             fontsize=12, fontweight="bold", color=INK, loc="left", pad=12)
save(fig, "Figure_1_System_Pipeline.pdf")

# ===== FIGURE 2 — Corpus construction flowchart =====
fig, ax = blank_ax(13, 5.4)
srcs = [("DravidianCodeMix", "~20K Ml-En comments\n→ real code-mixed ham"),
        ("Mishra & Soni", "5,971 tri-class\n638 smishing → seeds"),
        ("UCI SMS Spam", "5,574 English\nham / spam")]
for i, (t, sub) in enumerate(srcs):
    rbox(ax, 0.3, 3.9 - i*1.35, 3.0, 1.05, t, LIGHT, LINE, INK, fs=10, sub=sub)
    arrow(ax, 3.3, 4.42 - i*1.35, 4.1, 3.0)
rbox(ax, 4.1, 2.35, 2.0, 2.2, "Fuse &\nbalance", NAVY, NAVY, "white", fs=11, sub="cap ham 3,000", subc="#c7d2ec")
arrow(ax, 6.1, 3.45, 6.9, 4.2)
arrow(ax, 6.1, 3.45, 6.9, 2.1)
rbox(ax, 6.9, 3.65, 5.7, 1.1, "Persuasion-conditioned augmentation", BLUE_LT, BLUE, BLUE, fs=10,
     sub="Llama-3.3-70B · Cialdini principles · link-free patterns · 573 reviewed")
rbox(ax, 6.9, 1.55, 5.7, 1.0, "Dedup → stratified split (train-only synthetic)", BLUE_LT, BLUE, BLUE, fs=10,
     sub="exact + near-dup removal BEFORE splitting · 70/15/15")
ax.add_patch(FancyBboxPatch((0.3, 0.25), 12.3, 0.8, boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=1.0, edgecolor=LINE, facecolor=LIGHT))
ax.text(6.45, 0.65, "Final corpus:  5,174 messages  ·  3,034 ham · 1,399 smishing · 741 spam  ·  train 3,937 / val 618 / test 619",
        ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
ax.set_title("Figure 2.  Corpus construction — open data, theory-guided augmentation, leakage control",
             fontsize=12, fontweight="bold", color=INK, loc="left", pad=12)
save(fig, "Figure_2_Corpus_Construction.pdf")

# ===== FIGURE 3 — Research gap diagram =====
fig, ax = blank_ax(11, 6)
labels = [("D1", "SMS & smishing\ndetection"), ("D2", "Code-mixed\nIndic NLP"),
          ("D3", "LLM threat\ndetection"), ("D4", "On-device\nefficiency")]
pos = [(0.8, 4.3), (6.0, 4.3), (0.8, 2.9), (6.0, 2.9)]
for (tag, t), (x, y) in zip(labels, pos):
    rbox(ax, x, y, 4.2, 1.1, f"{tag} · {t.splitlines()[0]}", LIGHT, LINE, INK, fs=10.5,
         sub=t.splitlines()[1] if "\n" in t else None)
    arrow(ax, x+2.1, y, 5.5, 1.7, color=GREY)
rbox(ax, 3.0, 0.5, 5.0, 1.25, "THE GAP", NAVY, NAVY, "#8fabff", fs=12,
     sub="Tri-class, dual-script Malayalam-English smishing,\nIndic encoder, evaluated on-device", subc="white")
ax.text(5.5, 2.45, "integrated for the first time", ha="center", fontsize=10, color=BLUE, fontstyle="italic")
ax.set_title("Figure 3.  Four mature literatures and their unaddressed intersection",
             fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
save(fig, "Figure_3_Research_Gap.pdf")

# ===== FIGURE 4 — Model benchmark (grouped bars) =====
fig, ax = plt.subplots(figsize=(8.5, 5))
models = ["TF-IDF + RF\n(classical)", "MuRIL", "IndicBERT\n(deployed)"]
macro = [0.807, 0.805, 0.892]; smish = [0.645, 0.803, 0.868]
import numpy as np
x = np.arange(len(models)); bw = 0.36
b1 = ax.bar(x - bw/2, macro, bw, label="Macro-F1", color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
b2 = ax.bar(x + bw/2, smish, bw, label="Smishing recall", color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.012, f"{r.get_height():.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10.5)
ax.set_ylim(0, 1.0); ax.set_ylabel("Score (held-out test, Roman arm)")
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Figure 4.  Detector benchmark — IndicBERT leads on every metric",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=12)
save(fig, "Figure_4_Model_Benchmark.pdf")

# ===== FIGURE 5 — Key findings (before/after) =====
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
finds = [("Augmentation (G1)\nMuRIL smishing recall", 0.00, 0.803, GREEN, "up"),
         ("Quantization (G5)\nint8 smishing recall", 0.868, 0.368, RED, "down"),
         ("Script (G2)\nnative-script smishing recall", 0.868, 0.658, RED, "down")]
for ax, (title, a, b, col, dirn) in zip(axes, finds):
    bars = ax.bar(["before", "after"], [a, b], width=0.55, color=[GREY, col],
                  edgecolor="white", linewidth=1.5, zorder=3)
    for r, v in zip(bars, [a, b]):
        ax.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=INK)
    delta = b - a
    ax.text(0.5, 0.92, f"Δ {delta:+.3f}", transform=ax.transAxes, ha="center",
            fontsize=12, fontweight="bold", color=col)
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("smishing recall" if ax is axes[0] else "")
fig.suptitle("Figure 5.  Three findings — augmentation lifts, quantization & transliteration hurt the boundary class",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=1.02)
save(fig, "Figure_5_Key_Findings.pdf")

# ===== FIGURE 6 — Leakage validation (naive vs grouped) =====
fig, ax = plt.subplots(figsize=(8.5, 5))
groups = ["Word TF-IDF SVM", "Character 2–5-gram SVM"]
naive = [0.916, 0.921]; grouped = [0.885, 0.894]
x = np.arange(len(groups)); bw = 0.36
b1 = ax.bar(x - bw/2, naive, bw, label="Naïve split (leaky)", color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
b2 = ax.bar(x + bw/2, grouped, bw, label="Near-duplicate-grouped", color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.004, f"{r.get_height():.3f}",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
for i, (nv, gr) in enumerate(zip(naive, grouped)):
    ax.text(i, 0.86, f"−{nv-gr:.3f}", ha="center", fontsize=11, fontweight="bold", color=RED)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
ax.set_ylim(0.82, 0.94); ax.set_ylabel("Macro-F1 (5-seed mean)")
ax.legend(frameon=False, loc="lower center", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Figure 6.  Leakage inflates scores — grouping near-duplicates drops macro-F1",
             fontsize=12, fontweight="bold", color=INK, loc="left", pad=12)
save(fig, "Figure_6_Leakage_Validation.pdf")

# ===== FIGURE 7 — Corpus class distribution =====
fig, ax = plt.subplots(figsize=(8, 4.8))
cls = ["Ham", "Smishing", "Spam"]; counts = [3034, 1399, 741]; cols = [GREEN, RED, AMBER]
bars = ax.barh(cls[::-1], counts[::-1], color=cols[::-1], edgecolor="white", linewidth=2, zorder=3, height=0.62)
for r, v in zip(bars, counts[::-1]):
    ax.text(v+40, r.get_y()+r.get_height()/2, f"{v:,}", va="center", fontsize=12,
            fontweight="bold", color=INK)
ax.set_xlim(0, 3400); ax.set_xlabel("Messages (deduplicated corpus, n = 5,174)")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Figure 7.  Tri-class corpus composition — the smishing boundary class is synthesized",
             fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=12)
save(fig, "Figure_7_Corpus_Distribution.pdf")

print("\nAll figures ->", OUT)
