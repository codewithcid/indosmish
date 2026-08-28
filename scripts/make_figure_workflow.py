"""Figure 8 — IndoSmish training + on-device inference workflow (algorithm flowchart).
Classic flowchart symbols: process, decision, data cylinder, I/O parallelogram, terminator.
Run: python scripts/make_figure_workflow.py -> figures/Figure_8_Model_Workflow.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)

INK = "#222a38"; MUTED = "#5d646e"
PROC_F, PROC_E = "#d3e3fb", "#3f74c9"
DEC_F, DEC_E = "#fdeec2", "#d19b00"
CYL_F, CYL_E = "#d7efd9", "#3a9d5d"
IO_F, IO_E = "#fde2c4", "#df862f"
STOP_F, STOP_E = "#f9d2d2", "#d24a4a"
BAND_TRAIN = "#f5f7fb"; BAND_TEST = "#fbf8ef"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(9.2, 12.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")


def proc(cx, cy, w, h, text, fs=9.5, fill=PROC_F, ec=PROC_E, tc=INK, bold=True):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                linewidth=1.3, edgecolor=ec, facecolor=fill))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")


def decision(cx, cy, w, h, text, fs=9.5):
    ax.add_patch(Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                         closed=True, linewidth=1.3, edgecolor=DEC_E, facecolor=DEC_F))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold")


def cyl(cx, cy, w, h, text, fs=8.8):
    eh = h*0.16
    yb = cy - h/2 + eh      # bottom ellipse centre
    yt = cy + h/2 - eh      # top ellipse centre
    # bottom ellipse (full fill) — its lower half becomes the rounded bottom; upper half is covered by the body
    ax.add_patch(Ellipse((cx, yb), w, 2*eh, facecolor=CYL_F, edgecolor="none"))
    # body rectangle
    ax.add_patch(Rectangle((cx-w/2, yb), w, yt-yb, facecolor=CYL_F, edgecolor="none"))
    # side lines
    ax.plot([cx-w/2, cx-w/2], [yb, yt], color=CYL_E, lw=1.3)
    ax.plot([cx+w/2, cx+w/2], [yb, yt], color=CYL_E, lw=1.3)
    # bottom front arc (outline only)
    ax.add_patch(Arc((cx, yb), w, 2*eh, theta1=180, theta2=360, edgecolor=CYL_E, lw=1.3))
    # top ellipse (full, filled + outlined) drawn last so it sits on top
    ax.add_patch(Ellipse((cx, yt), w, 2*eh, facecolor=CYL_F, edgecolor=CYL_E, lw=1.3))
    ax.text(cx, cy - eh*0.2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold")


def io(cx, cy, w, h, text, fs=9):
    sk = min(w*0.14, 0.42)   # cap the slant so wide parallelograms keep usable text width
    ax.add_patch(Polygon([(cx-w/2+sk, cy+h/2), (cx+w/2, cy+h/2), (cx+w/2-sk, cy-h/2), (cx-w/2, cy-h/2)],
                         closed=True, linewidth=1.3, edgecolor=IO_E, facecolor=IO_F))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold")


def stop(cx, cy, r, text):
    ax.add_patch(Ellipse((cx, cy), r*2, r*1.5, linewidth=1.3, edgecolor=STOP_E, facecolor=STOP_F))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")


def arr(x1, y1, x2, y2, label=None, lx=0, ly=0, color="#54607a", fs=8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color, linestyle=ls, shrinkA=1, shrinkB=1))
    if label:
        ax.text((x1+x2)/2+lx, (y1+y2)/2+ly, label, ha="center", va="center", fontsize=fs,
                color=MUTED, fontstyle="italic")


# ---- phase bands ----
ax.add_patch(Rectangle((0.15, 6.15), 9.7, 8.6, facecolor=BAND_TRAIN, edgecolor="#dde3ee", lw=1))
ax.add_patch(Rectangle((0.15, 0.2), 9.7, 5.75, facecolor=BAND_TEST, edgecolor="#eadfc6", lw=1))
ax.text(0.45, 14.45, "TRAINING PHASE", fontsize=11, fontweight="bold", color="#3f74c9")
ax.text(0.45, 5.62, "INFERENCE / TEST PHASE  (on-device)", fontsize=11, fontweight="bold", color="#df862f")

CX = 4.4
LP = 7.15      # side-cylinder LEFT edge target (box right edge = 5.75, gap = 1.4)
CXC = 8.1      # side-cylinder centre (width 1.9 -> right edge 9.05, clears loop at 9.4)
LOOP = 9.4
# ---- training nodes ----
proc(CX, 14.0, 2.0, 0.7, "epoch = 0")
decision(CX, 12.75, 2.5, 1.15, "epoch < N ?")
stop(8.1, 12.75, 0.7, "stop")
proc(CX, 11.4, 2.7, 0.75, "Load IndicBERT +\nSentencePiece tokenizer")
cyl(CXC, 11.4, 1.9, 1.0, "Pretrained\nIndicBERT (HF)")
proc(CX, 10.2, 2.7, 0.75, "Tokenize batch\n(max_len 128)")
cyl(CXC, 10.2, 1.9, 1.0, "Training corpus\n(3,937 msgs)")
proc(CX, 9.0, 2.7, 0.75, "Fine-tune —\nclass-weighted loss")
proc(CX, 7.75, 2.9, 0.8, "Validate; keep best\nby macro-F1")
cyl(1.15, 7.95, 1.85, 1.05, "Best model")
proc(CX, 6.6, 2.0, 0.65, "epoch = epoch + 1")

# ---- training arrows ----
arr(CX, 13.65, CX, 13.33)
arr(CX+1.25, 12.75, 7.4, 12.75, "no", ly=0.2)
arr(CX, 12.18, CX, 11.78, "yes", lx=0.28)
arr(LP, 11.4, 5.75, 11.4, "load_pretrained()", ly=0.24, fs=7.5)
arr(CX, 11.02, CX, 10.58)
arr(LP, 10.2, 5.75, 10.2, "batch", ly=0.22, fs=8)
arr(CX, 9.82, CX, 9.38)
arr(CX, 8.62, CX, 8.15)
arr(CX-1.45, 7.75, 2.1, 7.75, "save_best()", ly=0.24, fs=7.5)
arr(CX, 7.35, CX, 6.93)
# loop back epoch+1 -> decision (enter from the top, far right clears the stop terminator)
ax.add_patch(FancyArrowPatch((CX+1.0, 6.6), (LOOP, 6.6), arrowstyle="-", linewidth=1.4, color="#54607a"))
ax.add_patch(FancyArrowPatch((LOOP, 6.6), (LOOP, 13.55), arrowstyle="-", linewidth=1.4, color="#54607a"))
ax.add_patch(FancyArrowPatch((LOOP, 13.55), (CX+0.05, 13.55), arrowstyle="-", linewidth=1.4, color="#54607a"))
arr(CX+0.05, 13.55, CX+0.05, 13.35)
ax.text(LOOP+0.2, 9.8, "next epoch", rotation=90, fontsize=8, color=MUTED, fontstyle="italic", va="center")

# ---- inference phase ----
proc(CX, 4.85, 2.9, 0.85, "Export best model →\nONNX (fp32)", fs=9.5)
proc(CX, 3.55, 2.9, 0.8, "IndicBERT (ONNX)\nin browser · WebAssembly")
cyl(CXC, 3.55, 1.9, 1.0, "Incoming SMS\n(on device)")
io(2.2, 2.35, 1.9, 0.72, "P(ham)")
io(4.4, 2.35, 1.9, 0.72, "P(spam)")
io(6.6, 2.35, 2.05, 0.72, "P(smishing)")
proc(CX, 1.25, 3.6, 0.65, "argmax  ·  max-value selector", fs=9)
io(CX, 0.5, 5.3, 0.68, "Predicted class:  ham / spam / smishing", fs=9)

# bridge best-model -> export (down the far left, then right into export)
ax.add_patch(FancyArrowPatch((1.15, 7.42), (1.15, 4.85), arrowstyle="-", linewidth=1.4, color="#54607a"))
arr(1.15, 4.85, CX-1.45, 4.85)
ax.text(0.98, 6.55, "load_best()", rotation=90, fontsize=7.5, color=MUTED, fontstyle="italic", va="center")
arr(CX, 4.42, CX, 3.98)
arr(LP, 3.55, 5.75, 3.55, "one SMS", ly=0.22, fs=8)
# detector -> three probabilities
arr(CX, 3.15, 2.2, 2.73)
arr(CX, 3.15, 4.4, 2.73)
arr(CX, 3.15, 6.6, 2.73)
# probs -> argmax
arr(2.2, 1.99, CX-0.6, 1.6)
arr(4.4, 1.99, CX, 1.6)
arr(6.6, 1.99, CX+0.6, 1.6)
arr(CX, 0.92, CX, 0.86)

# legend — placed in the clear lower-right of the inference band
lx, ly = 8.15, 2.35
ax.text(lx, ly + 0.42, "Legend", fontsize=8.5, fontweight="bold", color=INK)
for i, (f, e, lbl) in enumerate([(PROC_F, PROC_E, "Process"), (DEC_F, DEC_E, "Decision"),
                                  (CYL_F, CYL_E, "Data store"), (IO_F, IO_E, "I / O"), (STOP_F, STOP_E, "Terminator")]):
    ax.add_patch(Rectangle((lx, ly - i*0.4), 0.26, 0.2, facecolor=f, edgecolor=e, lw=1.2))
    ax.text(lx+0.36, ly - i*0.4 + 0.1, lbl, fontsize=8, va="center", color=INK)

ax.set_title("Figure 8.  IndoSmish training and on-device inference workflow",
             fontsize=13, fontweight="bold", color=INK, loc="left", pad=14)
fig.savefig(OUT / "Figure_8_Model_Workflow.pdf", bbox_inches="tight", pad_inches=0.2)
print("wrote Figure_8_Model_Workflow.pdf")
