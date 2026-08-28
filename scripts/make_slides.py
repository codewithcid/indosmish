"""Generate the IndoSmish capstone presentation (.pptx) — professional, detailed, 20 slides.

Run:  python scripts/make_slides.py
Output: IndoSmish_Presentation.pptx (project root)
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]

# ---- theme ----
BLUE = RGBColor(0x33, 0x66, 0xFF)
BLUE_DK = RGBColor(0x00, 0x43, 0xA9)
INK = RGBColor(0x24, 0x2C, 0x3A)
MUTED = RGBColor(0x5D, 0x64, 0x6E)
LIGHT = RGBColor(0xF4, 0xF6, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x00, 0xAC, 0x69)
AMBER = RGBColor(0xE6, 0xA9, 0x00)
RED = RGBColor(0xEC, 0x43, 0x33)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xE0, 0xE5, 0xF0)
HEAD = "Segoe UI Semibold"
BODY = "Segoe UI"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(1, 0, 0, SW, SH)
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background()
    bg.shadow.inherit = False
    s.shapes._spTree.remove(bg._element); s.shapes._spTree.insert(2, bg._element)
    return s


def box(s, x, y, w, h, fill=None, line=None, line_w=1.0):
    sp = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.shadow.inherit = False
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(line_w)
    return sp


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, sp_after=6):
    """runs: list of paragraphs; each paragraph = list of (txt, size, color, bold, font)."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(sp_after); p.space_before = Pt(0)
        for (txt, size, color, bold, font) in para:
            r = p.add_run(); r.text = txt
            r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
            r.font.name = font
    return tb


def R(t, s, c=INK, b=False, f=BODY):
    return (t, s, c, b, f)


def header(s, kicker, title):
    box(s, 0.55, 0.55, 0.14, 0.62, fill=BLUE)
    text(s, 0.85, 0.5, 11.8, 0.4, [[R(kicker.upper(), 12, BLUE, True, HEAD)]])
    text(s, 0.85, 0.82, 11.8, 0.9, [[R(title, 27, INK, True, HEAD)]])


def bullets(s, x, y, w, items, size=15, gap=10):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(5))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.space_before = Pt(0); p.line_spacing = 1.08
        lvl = it[2] if len(it) > 2 else 0
        dot = p.add_run(); dot.text = ("▸  " if lvl == 0 else "–  ")
        dot.font.size = Pt(size); dot.font.name = BODY
        dot.font.color.rgb = BLUE if lvl == 0 else MUTED
        head, rest = it[0], it[1]
        if head:
            rh = p.add_run(); rh.text = head + ("  " if rest else "")
            rh.font.size = Pt(size); rh.font.name = BODY; rh.font.bold = True; rh.font.color.rgb = INK
        if rest:
            rr = p.add_run(); rr.text = rest
            rr.font.size = Pt(size); rr.font.name = BODY; rr.font.color.rgb = MUTED
    return tb


def footer(s, n):
    text(s, 0.85, 7.05, 6, 0.3, [[R("IndoSmish · Capstone Project", 9, MUTED, False, BODY)]])
    text(s, 11.5, 7.05, 1.2, 0.3, [[R(str(n), 9, MUTED, False, BODY)]], align=PP_ALIGN.RIGHT)


# ===================== SLIDE 1 — TITLE =====================
s = slide()
box(s, 0, 0, 13.333, 7.5, fill=INK)
box(s, 0, 0, 13.333, 0.16, fill=BLUE)
# logo mark
box(s, 0.9, 0.9, 0.75, 0.75, fill=BLUE)
text(s, 0.9, 0.98, 0.75, 0.6, [[R("IS", 24, WHITE, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 1.85, 0.98, 8, 0.6, [[R("IndoSmish", 30, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.9, 2.7, 11.5, 2.0, [
    [R("On-Device Smishing Detection for", 40, WHITE, True, HEAD)],
    [R("Code-Mixed Malayalam–English SMS", 40, RGBColor(0x8F,0xAB,0xFF), True, HEAD)],
])
text(s, 0.95, 4.75, 11.4, 0.6, [[
    R("An implementation of the benchmark from ", 15, RGBColor(0xC9,0xD3,0xEA), False, BODY),
    R("“From SMS Spam Filters to Quantized LLMs”", 15, WHITE, True, BODY),
    R("  ·  IEEE Access, 2026", 15, RGBColor(0xC9,0xD3,0xEA), False, BODY),
]])
box(s, 0.9, 5.5, 11.5, 0.02, fill=RGBColor(0x3A,0x44,0x58))
text(s, 0.9, 5.7, 11.5, 1.0, [
    [R("Sidhardh S · Advaith Kamath · Joel G. Bert · T. Amarsainadh", 14, WHITE, False, BODY)],
    [R("School of Computer Science and Engineering (SCOPE), VIT-AP University", 12.5, RGBColor(0xA9,0xB4,0xCC), False, BODY)],
    [R("Live demo:  indosmish-cid.vercel.app", 12.5, RGBColor(0x8F,0xAB,0xFF), True, BODY)],
])

# ===================== SLIDE 2 — PROBLEM =====================
s = slide(); header(s, "Background", "The Problem: Smishing")
bullets(s, 0.85, 1.9, 7.2, [
    ("Smishing", "= phishing delivered over SMS: short, urgent, socially-engineered messages."),
    ("SMS is trusted", "— read within minutes, first-party interface → an attractive attack surface."),
    ("The economic toll is rising fast", "— ~$470M reported smishing losses in the US in 2024 (5× since 2020)."),
    ("Humans are unreliable", "— in a controlled study, people correctly identified only 67% of smishing."),
    ("Defenders lag", "— commercial anti-smishing tools trail live campaigns; generative AI lowers the cost of fluent, targeted lures."),
])
box(s, 8.5, 2.0, 4.0, 3.6, fill=LIGHT, line=LINE)
text(s, 8.85, 2.3, 3.4, 0.5, [[R("Why SMS is a hard substrate", 13, INK, True, HEAD)]])
bullets(s, 8.85, 2.85, 3.4, [
    ("160 characters", "→ heavy abbreviation", 1),
    ("No headers", "or routing metadata", 1),
    ("Sender IDs", "cheaply spoofed", 1),
    ("Casual register", "overlaps with real texts", 1),
], size=12.5, gap=8)
footer(s, 2)

# ===================== SLIDE 3 — WHY HARD IN INDIA =====================
s = slide(); header(s, "Motivation", "Why It's Harder in India")
bullets(s, 0.85, 1.9, 11.6, [
    ("Indian SMS is code-mixed", "— users write Malayalam / Tamil / etc. in Latin script, interleaved with English."),
    ("", ""),
])
box(s, 0.85, 2.65, 11.6, 0.85, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, line_w=1.2)
text(s, 1.1, 2.82, 11.1, 0.6, [[
    R("“ningalude account block aayi, ee link click cheyyu”", 17, BLUE_DK, True, "Consolas"),
    R("   — Romanized Malayalam + English", 13, MUTED, False, BODY),
]], anchor=MSO_ANCHOR.MIDDLE)
bullets(s, 0.85, 3.85, 11.6, [
    ("Non-standard orthography", "— the same word spelled many ways (jaldi / jldi / jldii); tokenizers fragment it."),
    ("Standard PLMs degrade", "on transliterated Hindi/Malayalam vs native script (documented deficit)."),
    ("Smishing is a boundary class", "— lexically closer to legitimate transactional messages than bulk spam is."),
    ("→ The real problem", "isn't just 'low-resource'; it's code-mixing + adversarial drift + privacy + limited hardware, together."),
])
footer(s, 3)

# ===================== SLIDE 4 — THE GAP =====================
s = slide(); header(s, "The Research Gap", "Four Literatures That Don't Touch")
cards = [
    ("SMS / smishing detection", "Rule-based → ML → DL → PLM → LLM, but ~80% English-only."),
    ("Code-mixed Indic NLP", "Transliteration, MuRIL, IndicBERT — but no security application."),
    ("LLM text-threat detection", "Prompting, augmentation, SLMs — concentrated on English email."),
    ("On-device efficiency", "Quantization, PEFT, edge profiling — not on security text."),
]
for i, (t, d) in enumerate(cards):
    x = 0.85 + (i % 2) * 6.05; y = 1.95 + (i // 2) * 1.35
    box(s, x, y, 5.8, 1.2, fill=LIGHT, line=LINE)
    box(s, x, y, 0.1, 1.2, fill=BLUE)
    text(s, x + 0.3, y + 0.16, 5.4, 0.4, [[R(t, 14, INK, True, HEAD)]])
    text(s, x + 0.3, y + 0.58, 5.4, 0.55, [[R(d, 12, MUTED, False, BODY)]])
box(s, 0.85, 4.85, 11.6, 1.4, fill=INK)
text(s, 1.15, 5.05, 11.1, 1.05, [
    [R("The intersection is empty.", 15, WHITE, True, HEAD)],
    [R("No public corpus separates smishing from spam in a code-mixed Indian language · no MuRIL/IndicBERT "
       "evaluated on Indian SMS threats · no quantized detector profiled on-device. ", 13, RGBColor(0xC9,0xD3,0xEA), False, BODY),
     R("Our capstone builds this missing benchmark.", 13, RGBColor(0x8F,0xAB,0xFF), True, BODY)],
])
footer(s, 4)

# ===================== SLIDE 5 — OBJECTIVES =====================
s = slide(); header(s, "Scope", "Objectives")
items = [
    ("Corpus", "Build a tri-class (ham / spam / smishing) code-mixed Malayalam–English SMS corpus."),
    ("Baselines", "Establish Indic encoder baselines — classical, MuRIL, IndicBERT."),
    ("Quantization", "Measure how int8 quantization affects the security-critical smishing class."),
    ("Script", "Test the transliteration deficit — same messages in Roman vs native script."),
    ("Deploy", "Ship an on-device, privacy-preserving detector — entirely on free infrastructure."),
]
for i, (t, d) in enumerate(items):
    y = 1.95 + i * 0.92
    c = box(s, 0.85, y, 1.7, 0.72, fill=BLUE)
    text(s, 0.85, y, 1.7, 0.72, [[R(t, 13, WHITE, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 2.75, y, 9.7, 0.72, [[R(d, 15, INK, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
footer(s, 5)

# ===================== SLIDE 6 — ARCHITECTURE =====================
s = slide(); header(s, "System", "Architecture — Fully On-Device")
steps = [("1", "Input SMS", "code-mixed text"), ("2", "Tokenize", "SentencePiece 200K"),
         ("3", "IndicBERT", "ONNX on WASM"), ("4", "Softmax", "3-way probability"),
         ("5", "Verdict", "label + signals")]
n = len(steps); w = 2.15; gap = 0.28; total = n * w + (n - 1) * gap
x0 = (13.333 - total) / 2
for i, (num, t, d) in enumerate(steps):
    x = x0 + i * (w + gap)
    box(s, x, 2.4, w, 1.5, fill=LIGHT, line=LINE)
    cc = box(s, x + w/2 - 0.22, 2.2, 0.44, 0.44, fill=BLUE)
    text(s, x + w/2 - 0.22, 2.2, 0.44, 0.44, [[R(num, 14, WHITE, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, 2.95, w, 0.4, [[R(t, 14, INK, True, HEAD)]], align=PP_ALIGN.CENTER)
    text(s, x, 3.38, w, 0.4, [[R(d, 11, MUTED, False, BODY)]], align=PP_ALIGN.CENTER)
    if i < n - 1:
        text(s, x + w - 0.02, 2.4, gap + 0.1, 1.5, [[R("→", 20, BLUE, True, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.85, 4.5, 11.6, 1.5, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, line_w=1.2)
text(s, 1.15, 4.72, 11.1, 1.1, [
    [R("Every step runs client-side.", 15, BLUE_DK, True, HEAD)],
    [R("The model is fetched once, then inference happens in the browser via transformers.js on WebAssembly. "
       "No backend, no cold start — and the SMS never leaves the device, realising the paper's on-device "
       "privacy thesis literally.  ≈ 15 ms per message on CPU.", 13, INK, False, BODY)],
])
footer(s, 6)

# ===================== SLIDE 7 — DATASETS =====================
s = slide(); header(s, "Data", "Datasets & Sources")
rows = [
    ("DravidianCodeMix (Ml–En)", "HuggingFace", "~20,000 comments", "Real code-mixed register → ham + style grounding"),
    ("Mishra & Soni tri-class", "Mendeley", "5,971 messages", "ham 4,844 / spam 489 / smishing 638 → smishing seeds"),
    ("UCI SMS Spam Collection", "UCI / HF", "5,574 messages", "English ham / spam → volume + English coverage"),
]
box(s, 0.85, 1.95, 11.6, 0.5, fill=INK)
for c, (lbl, wpx) in enumerate([("Dataset", 3.4), ("Source", 1.7), ("Size", 2.2), ("Role in the corpus", 4.3)]):
    xs = 0.85 + sum([3.4,1.7,2.2][:c]) if c else 0.85
text(s, 1.05, 1.98, 3.2, 0.44, [[R("Dataset", 12, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
text(s, 4.25, 1.98, 1.6, 0.44, [[R("Source", 12, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
text(s, 5.95, 1.98, 2.2, 0.44, [[R("Size", 12, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
text(s, 8.2, 1.98, 4.1, 0.44, [[R("Role", 12, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
for i, (d, src, sz, role) in enumerate(rows):
    y = 2.45 + i * 0.72
    box(s, 0.85, y, 11.6, 0.72, fill=WHITE if i % 2 else LIGHT, line=LINE)
    text(s, 1.05, y, 3.2, 0.72, [[R(d, 12.5, INK, True, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 4.25, y, 1.6, 0.72, [[R(src, 11.5, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 5.95, y, 2.2, 0.72, [[R(sz, 11.5, INK, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.2, y, 4.1, 0.72, [[R(role, 11, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.85, 4.85, 11.6, 0.9, [[
    R("Key insight:  ", 13, BLUE, True, HEAD),
    R("no public corpus separates smishing from spam in a code-mixed Indian language — so the smishing class "
      "itself is synthesized (next slide), grounded in real code-mixed text and real fraud pretexts.", 13, INK, False, BODY),
]])
footer(s, 7)

# ===================== SLIDE 8 — BUILDING THE CORPUS =====================
s = slide(); header(s, "Data", "Building the Corpus")
bullets(s, 0.85, 1.95, 6.9, [
    ("Fuse the three sources", "into one unified tri-class schema."),
    ("Balance", "— cap ham at 3,000; keep all native-script code-mixed ham."),
    ("Deduplicate BEFORE splitting", "— exact + near-duplicate signatures, to prevent train/test leakage."),
    ("Stratified 70 / 15 / 15 split", "with fixed seed 42."),
    ("Synthetic data is train-only", "— validation & test contain only human messages."),
])
# stat panel
box(s, 8.1, 1.95, 4.35, 3.9, fill=LIGHT, line=LINE)
text(s, 8.4, 2.15, 3.8, 0.4, [[R("FINAL CORPUS", 12, BLUE, True, HEAD)]])
stats = [("5,174", "messages (deduplicated)", INK),
         ("3,034", "ham", GREEN), ("1,399", "smishing", RED), ("741", "spam", AMBER)]
for i, (big, lbl, col) in enumerate(stats):
    y = 2.6 + i * 0.72
    text(s, 8.4, y, 1.7, 0.6, [[R(big, 24, col, True, HEAD)]])
    text(s, 10.2, y, 2.1, 0.6, [[R(lbl, 12.5, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
box(s, 8.4, 5.5, 3.75, 0.02, fill=LINE)
text(s, 8.4, 5.55, 3.8, 0.3, [[R("Splits:  train 3,937  ·  val 618  ·  test 619", 11.5, INK, True, BODY)]])
footer(s, 8)

# ===================== SLIDE 9 — AUGMENTATION =====================
s = slide(); header(s, "Core Contribution", "Data Augmentation")
bullets(s, 0.85, 1.85, 11.6, [
    ("The problem", "— code-mixed smishing doesn't exist in any dataset, so we synthesize it, faithfully."),
    ("Persuasion-conditioned generation", "— each message is grounded in a Cialdini persuasion principle "
     "(authority, scarcity, urgency, liking, reciprocity, social-proof), preserving the social-engineering mechanics."),
    ("Generator", "— Llama-3.3-70B via the Groq API (free); 750 link-based variants → 573 human-reviewed."),
    ("Link-free scam patterns", "— 14 archetypes (impersonation, callback, family-emergency, OTP-social, job, "
     "refund, loan, investment, romance…) in code-mixed AND English registers."),
    ("Ham hard-negatives", "— 156 legitimate look-alikes (real OTPs, bank alerts, deliveries, “send me 200 for "
     "lunch”) so the model learns that numbers + an action ≠ fraud; intent does."),
])
footer(s, 9)

# ===================== SLIDE 10 — PROTOCOL =====================
s = slide(); header(s, "Rigour", "Evaluation Protocol (Frozen)")
bullets(s, 0.85, 1.95, 11.6, [
    ("Primary metric: macro-F1", "— always reported with per-class smishing recall (the minority class that matters)."),
    ("Deduplicate before splitting", "— avoids the near-duplicate leakage that inflated older SMS benchmarks."),
    ("Script as a controlled variable", "— Roman / native Malayalam / transliterated evaluated separately."),
    ("Synthetic is train-only", "— val/test are human messages; no synthetic message or its near-duplicate leaks in."),
    ("No pooled averaging", "— binary and tri-class results are not exchangeable; contamination is disclosed."),
])
box(s, 0.85, 5.7, 11.6, 0.7, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, line_w=1.0)
text(s, 1.1, 5.72, 11.1, 0.66, [[
    R("The protocol was frozen before any model was trained", 12.5, BLUE_DK, True, BODY),
    R(" — so results are honest and reproducible, not tuned to the test set.", 12.5, INK, False, BODY),
]], anchor=MSO_ANCHOR.MIDDLE)
footer(s, 10)

# ===================== SLIDE 11 — MODELS =====================
s = slide(); header(s, "Models", "Baselines & Encoders")
mods = [
    ("TF-IDF + Random Forest", "Classical floor", "Character n-grams, class-balanced. Tiny, interpretable, CPU-only."),
    ("MuRIL", "Indic encoder", "Multilingual — 16 Indian languages + English + transliterated counterparts."),
    ("IndicBERT", "Indic encoder (deployed)", "Compact ALBERT-style · 200K SentencePiece vocab · ~135 MB · best accuracy."),
]
for i, (t, tag, d) in enumerate(mods):
    y = 1.95 + i * 1.28
    box(s, 0.85, y, 11.6, 1.15, fill=LIGHT, line=LINE)
    box(s, 0.85, y, 0.1, 1.15, fill=BLUE)
    text(s, 1.15, y + 0.13, 6, 0.4, [[R(t, 16, INK, True, HEAD), R("    " + tag, 11.5, BLUE, True, BODY)]])
    text(s, 1.15, y + 0.58, 11, 0.5, [[R(d, 13, MUTED, False, BODY)]])
text(s, 0.85, 5.95, 11.6, 0.6, [[
    R("Fine-tuned with class-weighted cross-entropy", 13, BLUE, True, HEAD),
    R("  to stop the minority smishing class collapsing into spam. Encoders trained locally on an RTX 3050 (4 GB).", 13, INK, False, BODY),
]])
footer(s, 11)

# ===================== SLIDE 12 — RESULTS TABLE =====================
s = slide(); header(s, "Results", "Benchmark — Held-Out Test Set")
box(s, 1.6, 2.1, 10.1, 0.6, fill=INK)
cols = [("Model", 1.9, 4.2), ("Type", 6.1, 2.4), ("Macro-F1", 8.5, 1.6), ("Smishing recall", 10.1, 1.6)]
text(s, 1.9, 2.12, 4.0, 0.56, [[R("Model", 13, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
text(s, 6.1, 2.12, 2.2, 0.56, [[R("Type", 13, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
text(s, 8.3, 2.12, 1.7, 0.56, [[R("Macro-F1", 13, WHITE, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 10.0, 2.12, 1.7, 0.56, [[R("Smishing recall", 12, WHITE, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
data = [("TF-IDF + Random Forest", "Classical", "0.807", "0.645", False),
        ("MuRIL", "Indic encoder", "0.805", "0.803", False),
        ("IndicBERT", "Indic encoder", "0.892", "0.868", True)]
for i, (m, t, f1, sr, best) in enumerate(data):
    y = 2.7 + i * 0.66
    box(s, 1.6, y, 10.1, 0.66, fill=RGBColor(0xEF,0xF3,0xFF) if best else (WHITE if i % 2 else LIGHT), line=LINE)
    mc = BLUE_DK if best else INK
    nm = m + ("   ● deployed" if best else "")
    text(s, 1.9, y, 4.2, 0.66, [[R(nm, 13.5, mc, best, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.1, y, 2.2, 0.66, [[R(t, 12, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.3, y, 1.7, 0.66, [[R(f1, 14, mc, best, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.0, y, 1.7, 0.66, [[R(sr, 14, mc, best, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 1.6, 5.1, 10.1, 0.8, [[
    R("IndicBERT wins on every metric — and is the smallest model, ", 14, INK, False, BODY),
    R("making the most accurate encoder also the most deployable.", 14, BLUE, True, BODY),
]], align=PP_ALIGN.CENTER)
footer(s, 12)

# ===================== SLIDES 13-15 — FINDINGS =====================
def finding_slide(n, tag, title, frm, to, up, desc):
    s = slide(); header(s, "Key Finding · " + tag, title)
    box(s, 0.85, 2.4, 5.6, 2.7, fill=LIGHT, line=LINE)
    text(s, 0.85, 2.75, 5.6, 0.4, [[R("SMISHING RECALL", 13, MUTED, True, HEAD)]], align=PP_ALIGN.CENTER)
    text(s, 0.85, 3.25, 2.7, 1.4, [[R(frm, 46, MUTED, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 3.4, 3.25, 0.6, 1.4, [[R("→", 30, INK, True, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 3.75, 3.25, 2.7, 1.4, [[R(to, 54, GREEN if up else RED, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.9, 2.6, 5.5, 3.2, [[R(d0, 15, INK, b0, BODY)] for (d0, b0) in desc], anchor=MSO_ANCHOR.MIDDLE, sp_after=12)
    footer(s, n)

finding_slide(13, "Augmentation (G1)", "Code-Mixed Data Unlocks the Boundary Class",
    "0.00", "0.803", True,
    [("MuRIL, before vs after training on the augmented code-mixed corpus.", True),
     ("Without code-mixed data, every smishing message collapses into spam (0% recall).", False),
     ("The missing ingredient is data, not model capacity — the survey's central claim, demonstrated.", False)])

finding_slide(14, "Quantization (G5)", "int8 Disproportionately Harms Fraud Detection",
    "0.868", "0.368", False,
    [("IndicBERT under dynamic int8 quantization.", True),
     ("The collapse holds across ALL configs (per-channel, avx512, reduce-range) — while ham stays perfect.", False),
     ("Quantization hurts the security-critical minority class most → we ship at full precision (fp32).", False)])

finding_slide(15, "Script (G2)", "A Measurable Transliteration Deficit",
    "0.868", "0.658", False,
    [("Same messages, Roman → native Malayalam script (macro-F1 also drops 0.892 → 0.803).", True),
     ("A 21-point drop confirms the deficit reported for transliterated Indic text (Krishnan et al.).", False),
     ("Native arm built via LLM transliteration — the paper's strategy 4.", False)])

# ===================== SLIDE 16 — DEPLOYMENT =====================
s = slide(); header(s, "Deployment", "On-Device Inference in the Browser")
bullets(s, 0.85, 1.95, 7.0, [
    ("IndicBERT → ONNX → transformers.js", "running on WebAssembly, entirely client-side."),
    ("No backend, no cold start", "— free to host and run, forever."),
    ("Privacy by construction", "— the SMS never leaves the user's device."),
    ("~15 ms per message on CPU", "— on-device-grade latency."),
    ("Live on Vercel + GitHub Pages", "as a polished, professional web app."),
])
box(s, 8.2, 2.0, 4.3, 3.6, fill=INK)
text(s, 8.55, 2.3, 3.7, 0.4, [[R("HOW WE GOT HERE", 11, RGBColor(0x8F,0xAB,0xFF), True, HEAD)]])
_dt = s.shapes.add_textbox(Inches(8.55), Inches(2.85), Inches(3.65), Inches(2.5))
_tf = _dt.text_frame; _tf.word_wrap = True
_tf.margin_left = _tf.margin_right = _tf.margin_top = _tf.margin_bottom = 0
_dark = [
    ("Render free (512 MB)", "OOM'd loading the 128 MB model"),
    ("HF Docker Spaces", "now require a PRO plan"),
    ("Pivoted to browser inference", "— the model runs client-side"),
    ("A constraint became", "the on-device privacy story"),
]
for _i, (_h, _d) in enumerate(_dark):
    _p = _tf.paragraphs[0] if _i == 0 else _tf.add_paragraph()
    _p.space_after = Pt(12); _p.line_spacing = 1.05
    _r = _p.add_run(); _r.text = "—  "; _r.font.size = Pt(12); _r.font.name = BODY; _r.font.color.rgb = RGBColor(0x8F,0xAB,0xFF)
    _r = _p.add_run(); _r.text = _h; _r.font.size = Pt(12); _r.font.name = BODY; _r.font.bold = True; _r.font.color.rgb = WHITE
    _r = _p.add_run(); _r.text = "  " + _d; _r.font.size = Pt(12); _r.font.name = BODY; _r.font.color.rgb = RGBColor(0xB4,0xBE,0xD4)
footer(s, 16)

# ===================== SLIDE 17 — ENGINEERING =====================
s = slide(); header(s, "Engineering", "Building It All on Free Infrastructure")
grid = [
    ("Hardware", "RTX 3050 (4 GB) for encoders locally; free Kaggle T4 for the quantized SLM track."),
    ("Data generation", "Groq free tier (Llama-3.3-70B) for persuasion-conditioned augmentation & transliteration."),
    ("Model hosting", "Public HuggingFace model repo; the browser fetches it once and caches it."),
    ("Iterative hardening", "Added link-free scam patterns + ham hard-negatives after real-world testing."),
    ("Toolchain", "transformers, onnxruntime, tokenizers, scikit-learn, python-pptx — all open source."),
    ("Total cost", "₹0 — every component runs on a free tier."),
]
for i, (t, d) in enumerate(grid):
    x = 0.85 + (i % 2) * 6.05; y = 1.95 + (i // 2) * 1.35
    box(s, x, y, 5.8, 1.2, fill=LIGHT, line=LINE)
    text(s, x + 0.28, y + 0.15, 5.3, 0.4, [[R(t, 14, BLUE, True, HEAD)]])
    text(s, x + 0.28, y + 0.55, 5.3, 0.6, [[R(d, 11.5, MUTED, False, BODY)]])
footer(s, 17)

# ===================== SLIDE 18 — LIVE DEMO =====================
s = slide(); header(s, "Demo", "Try It Live")
box(s, 0.85, 1.95, 11.6, 2.0, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, line_w=1.2)
text(s, 1.2, 2.2, 11.0, 0.6, [[R("indosmish-cid.vercel.app", 26, BLUE_DK, True, HEAD)]])
text(s, 1.2, 2.95, 11.0, 0.9, [[
    R("Paste any SMS → a tri-class verdict with class probabilities, the detected persuasion signals, "
      "and on-device latency. Everything computes in your browser.", 14, INK, False, BODY)]])
text(s, 0.85, 4.3, 11.6, 0.4, [[R("SAMPLE INPUTS", 12, MUTED, True, HEAD)]])
samples = [('"Hey da, reaching home by 8, dinner undo?"', "→ HAM", GREEN),
           ('"ningalude SBI account block aayi, ee link click cheyyu…"', "→ SMISHING", RED),
           ('"FLAT 50% OFF this weekend only! Visit our store"', "→ SPAM", AMBER)]
for i, (msg, verd, col) in enumerate(samples):
    y = 4.75 + i * 0.62
    box(s, 0.85, y, 11.6, 0.55, fill=WHITE, line=LINE)
    text(s, 1.1, y, 8.6, 0.55, [[R(msg, 13, INK, False, "Consolas")]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 9.9, y, 2.4, 0.55, [[R(verd, 13, col, True, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
footer(s, 18)

# ===================== SLIDE 19 — LIMITATIONS =====================
s = slide(); header(s, "Reflection", "Limitations & Future Work")
bullets(s, 0.85, 1.95, 11.6, [
    ("Quantized SLM track (S4–S5)", "— Qwen2.5-1.5B zero/few-shot + 4-bit QLoRA is prepared as a one-click "
     "Kaggle notebook (needs a GPU we don't have locally)."),
    ("Ambiguous conversational scams", "remain genuinely hard — 'send money to this number' overlaps with real "
     "messages; a precision/recall trade-off, not a bug."),
    ("Native-script arm", "would benefit from a dedicated native corpus, not only LLM transliteration."),
    ("First-load download", "— the full-precision model is 128 MB (one-time, then cached); kept at fp32 to preserve accuracy."),
    ("Broader languages", "— the pipeline generalizes to Tamil/Telugu/Hindi code-mix as a next step."),
])
footer(s, 19)

# ===================== SLIDE 20 — CONCLUSION =====================
s = slide()
box(s, 0, 0, 13.333, 7.5, fill=INK)
box(s, 0, 0, 13.333, 0.16, fill=BLUE)
text(s, 0.9, 0.9, 11.5, 0.9, [[R("Conclusion", 34, WHITE, True, HEAD)]])
bullets_dark = [
    ("Delivered the paper's missing benchmark", "— corpus, baselines, three findings, and a live demo."),
    ("Three findings", "map directly to gaps G1 (augmentation), G2 (transliteration), G5 (quantization)."),
    ("IndicBERT — 0.892 macro-F1", "deployed on-device, privacy-preserving, and free."),
]
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(3))
tf = tb.text_frame; tf.word_wrap = True
for i, (h, d) in enumerate(bullets_dark):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(16); p.line_spacing = 1.1
    r = p.add_run(); r.text = "▸  "; r.font.color.rgb = BLUE; r.font.size = Pt(17); r.font.name = BODY
    r = p.add_run(); r.text = h + "  "; r.font.bold = True; r.font.color.rgb = WHITE; r.font.size = Pt(17); r.font.name = BODY
    r = p.add_run(); r.text = d; r.font.color.rgb = RGBColor(0xC9,0xD3,0xEA); r.font.size = Pt(16); r.font.name = BODY
box(s, 0.9, 5.2, 11.5, 0.02, fill=RGBColor(0x3A,0x44,0x58))
text(s, 0.9, 5.45, 11.5, 1.2, [
    [R("Live demo:  ", 15, RGBColor(0x8F,0xAB,0xFF), True, BODY), R("indosmish-cid.vercel.app", 15, WHITE, True, BODY)],
    [R("Based on:  ", 13, RGBColor(0x8F,0xAB,0xFF), True, BODY),
     R("“From SMS Spam Filters to Quantized LLMs” · IEEE Access 2026 · VIT-AP University", 13, RGBColor(0xC9,0xD3,0xEA), False, BODY)],
    [R("Thank you.", 20, WHITE, True, HEAD)],
])

out = ROOT / "IndoSmish_Presentation.pptx"
prs.save(str(out))
print(f"Saved {len(prs.slides._sldIdLst)} slides -> {out}")
