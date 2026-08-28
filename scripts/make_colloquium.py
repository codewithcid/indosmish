"""IndoSmish colloquium-format presentation (.pptx).
Structure: Title, Contents, Introduction, Literature Survey (table), Research Gap
(2-col), Problem Statement, Objectives, Proposed Methodology (flowcharts), Results &
Discussion (protocol validation + implementation), Conclusion.

Run:  python scripts/make_colloquium.py  ->  IndoSmish_Colloquium.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]

NAVY = RGBColor(0x1B, 0x35, 0x6B)      # header bar
BLUE = RGBColor(0x33, 0x66, 0xFF)      # accent
BLUE_DK = RGBColor(0x00, 0x43, 0xA9)
INK = RGBColor(0x22, 0x2A, 0x38)
MUTED = RGBColor(0x5D, 0x64, 0x6E)
LIGHT = RGBColor(0xF3, 0xF6, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1F, 0xA8, 0x6B)
AMBER = RGBColor(0xD9, 0x93, 0x00)
RED = RGBColor(0xD8, 0x3A, 0x4A)
LINE = RGBColor(0xD9, 0xE0, 0xEE)
HEAD = "Segoe UI Semibold"
BODY = "Segoe UI"

prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
PAGE = [0]


def slide(white=True):
    s = prs.slides.add_slide(BLANK)
    if white:
        bg = s.shapes.add_shape(1, 0, 0, SW, SH)
        bg.fill.solid(); bg.fill.fore_color.rgb = WHITE; bg.line.fill.background(); bg.shadow.inherit = False
        s.shapes._spTree.remove(bg._element); s.shapes._spTree.insert(2, bg._element)
    return s


def box(s, x, y, w, h, fill=None, line=None, lw=1.0, rounded=False):
    sp = s.shapes.add_shape(5 if rounded else 1, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.shadow.inherit = False
    if rounded:
        try: sp.adjustments[0] = 0.08
        except Exception: pass
    if fill is None: sp.fill.background()
    else: sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None: sp.line.fill.background()
    else: sp.line.color.rgb = line; sp.line.width = Pt(lw)
    return sp


def txt(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, sp=5, ls=1.05):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(sp); p.space_before = Pt(0); p.line_spacing = ls
        for (t, sz, c, b, f) in para:
            r = p.add_run(); r.text = t; r.font.size = Pt(sz); r.font.color.rgb = c; r.font.bold = b; r.font.name = f
    return tb


def R(t, sz, c=INK, b=False, f=BODY): return (t, sz, c, b, f)


def header(s, title, contd=False):
    PAGE[0] += 1
    box(s, 0, 0, 13.333, 1.02, fill=NAVY)
    box(s, 0, 1.02, 13.333, 0.06, fill=BLUE)
    txt(s, 0.55, 0, 11.0, 1.02, [[R(title + ("  (contd.)" if contd else ""), 26, WHITE, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
    txt(s, 12.2, 0, 0.9, 1.02, [[R(str(PAGE[0]), 15, RGBColor(0xB9,0xC6,0xE6), False, BODY)]], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def foot(s):
    txt(s, 0.55, 7.08, 8, 0.3, [[R("IndoSmish · Smishing Detection Beyond English Benchmarks · VIT-AP University", 8.5, MUTED, False, BODY)]])


def bullets(s, x, y, w, items, size=15, gap=10, ls=1.12):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(5.4))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        lvl = it[2] if len(it) > 2 else 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.space_before = Pt(0); p.line_spacing = ls
        d = p.add_run(); d.text = ("▪  " if lvl == 0 else "–  ")
        d.font.size = Pt(size); d.font.name = BODY; d.font.color.rgb = BLUE if lvl == 0 else MUTED
        if it[0]:
            h = p.add_run(); h.text = it[0] + ("  " if it[1] else "")
            h.font.size = Pt(size); h.font.name = BODY; h.font.bold = True; h.font.color.rgb = INK
        if it[1]:
            r = p.add_run(); r.text = it[1]; r.font.size = Pt(size); r.font.name = BODY; r.font.color.rgb = MUTED
    return tb


def table(s, x, y, w, col_w, rows, header_fill=NAVY, fs=10.5, hfs=10.5, rh=0.34):
    n = len(rows); ncol = len(col_w)
    gt = s.shapes.add_table(n, ncol, Inches(x), Inches(y), Inches(w), Inches(rh * n)).table
    gt.first_row = False; gt.horz_banding = False
    # remove default style banding
    tblPr = gt._tbl.tblPr
    for c in list(tblPr):
        tblPr.remove(c)
    for j, cw in enumerate(col_w):
        gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = gt.cell(i, j)
            c.margin_left = Inches(0.07); c.margin_right = Inches(0.05)
            c.margin_top = Inches(0.02); c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else (LIGHT if i % 2 else WHITE)
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.line_spacing = 1.0
            r = p.add_run(); r.text = str(cell)
            r.font.size = Pt(hfs if i == 0 else fs); r.font.name = BODY
            r.font.bold = (i == 0)
            r.font.color.rgb = WHITE if i == 0 else INK
    return gt


def fbox(s, x, y, w, h, title, sub, fill=LIGHT, tcol=INK, line=LINE):
    box(s, x, y, w, h, fill=fill, line=line, rounded=True)
    txt(s, x + 0.12, y + 0.08, w - 0.24, 0.4, [[R(title, 12, tcol, True, HEAD)]], align=PP_ALIGN.CENTER)
    txt(s, x + 0.12, y + 0.42, w - 0.24, h - 0.5, [[R(sub, 9.5, MUTED, False, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP)


def arrow(s, x, y, w, h=0.4, char="→"):
    txt(s, x, y, w, h, [[R(char, 20, BLUE, True, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ============ 1. TITLE ============
s = slide(white=False)
box(s, 0, 0, 13.333, 7.5, fill=NAVY)
box(s, 0, 0, 13.333, 0.18, fill=BLUE)
box(s, 0, 2.15, 13.333, 0.02, fill=RGBColor(0x35,0x4B,0x82))
txt(s, 0.9, 0.7, 11.5, 1.2, [
    [R("SMISHING DETECTION BEYOND ENGLISH BENCHMARKS", 13, RGBColor(0x9F,0xB0,0xDA), True, HEAD)],
], sp=2)
txt(s, 0.9, 2.5, 11.6, 2.2, [
    [R("IndoSmish:", 34, WHITE, True, HEAD)],
    [R("On-Device Detection of Code-Mixed", 30, WHITE, True, HEAD)],
    [R("Malayalam–English Smishing", 30, RGBColor(0x8F,0xAB,0xFF), True, HEAD)],
], sp=4)
txt(s, 0.95, 5.15, 11.4, 0.5, [[R("A Critical Review of Code-Mixed Indic Text and On-Device Language Models", 14, RGBColor(0xC7,0xD2,0xEC), False, BODY)]])
box(s, 0.9, 5.7, 11.5, 0.02, fill=RGBColor(0x35,0x4B,0x82))
txt(s, 0.9, 5.9, 8.5, 1.3, [
    [R("Presented by:  Sidhardh S · Advaith Kamath · Joel G. Bert · T. Amarsainadh", 13, WHITE, False, BODY)],
    [R("Guides:  Dr. Bileesh P. Babu · Dr. Gokul Yenduri", 12.5, RGBColor(0xC7,0xD2,0xEC), False, BODY)],
    [R("School of Computer Science and Engineering (SCOPE), VIT-AP University", 11.5, RGBColor(0x9F,0xB0,0xDA), False, BODY)],
], sp=6)
txt(s, 9.7, 5.9, 2.9, 1.0, [[R("Live demo", 11, RGBColor(0x9F,0xB0,0xDA), True, HEAD)], [R("indosmish-cid.vercel.app", 12, WHITE, True, BODY)]], align=PP_ALIGN.RIGHT, sp=3)

# ============ 2. CONTENTS ============
s = slide(); header(s, "Contents")
items = ["Introduction", "Literature Survey", "Research Gap", "Problem Statement",
         "Objectives", "Proposed Methodology", "Results & Discussion", "Conclusion & Future Work"]
for i, it in enumerate(items):
    col = i // 4; row = i % 4
    x = 0.9 + col * 6.2; y = 1.6 + row * 1.2
    box(s, x, y, 0.6, 0.6, fill=BLUE, rounded=True)
    txt(s, x, y, 0.6, 0.6, [[R(f"{i+1:02d}", 16, WHITE, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, x + 0.85, y, 5.1, 0.6, [[R(it, 17, INK, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
foot(s)

# ============ 3. INTRODUCTION ============
s = slide(); header(s, "Introduction")
bullets(s, 0.6, 1.35, 7.4, [
    ("Smishing", "= phishing delivered over SMS. A lure uses an urgent pretext and asks the user to open a link, call a number, reply, share a credential, or transfer money."),
    ("Judged from very little text", "— often before any reputation service has seen the sender or destination."),
    ("Rising economic toll", "— the U.S. FTC reports ~$470 million in 2024 losses from scams that began with a text message."),
    ("Humans are unreliable filters", "— in a 187-participant study, only 67.1% of smishing was identified correctly."),
    ("Attackers adapt fast", "— shared infrastructure, high-volume delivery, and generative-AI-written lures."),
])
box(s, 8.35, 1.5, 4.35, 4.6, fill=LIGHT, line=LINE, rounded=True)
txt(s, 8.6, 1.7, 3.9, 0.4, [[R("THREE CLASSES TO SEPARATE", 11, BLUE, True, HEAD)]])
for i, (t, d, c) in enumerate([("Ham", "normal / personal / transactional", GREEN),
                                ("Spam", "bulk, promotional, not malicious", AMBER),
                                ("Smishing", "fraud: pretext + persuasion + action vector", RED)]):
    y = 2.25 + i * 1.15
    box(s, 8.6, y, 0.14, 0.9, fill=c)
    txt(s, 8.9, y, 3.6, 0.9, [[R(t, 15, c, True, HEAD)], [R(d, 11, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE, sp=2)
foot(s)

# ============ 4. INTRODUCTION contd — Indian register ============
s = slide(); header(s, "Introduction", contd=True)
txt(s, 0.6, 1.3, 12.1, 0.5, [[R("India makes the language and deployment constraints especially visible.", 16, INK, True, HEAD)]])
box(s, 0.6, 1.95, 12.1, 0.9, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, lw=1.2, rounded=True)
txt(s, 0.9, 2.12, 11.5, 0.6, [[
    R("“ningalude account block aayi, ee link click cheyyu”", 17, BLUE_DK, True, "Consolas"),
    R("   — Romanized Malayalam + English, one message", 12, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
bullets(s, 0.6, 3.15, 12.1, [
    ("Code-mixing + Romanization", "— Indic languages written in Latin script, mixed with English within a sentence or even a word."),
    ("Unstandardised spelling", "— the same word appears as “jaldi / jldi / jldii”; tokenizers fragment Romanized Indic words."),
    ("Documented PLM deficit", "— multilingual encoders underperform on transliterated Hindi & Malayalam vs native script."),
    ("Benchmark–deployment mismatch", "— most SMS filtering still uses an English corpus from 2011; the deployment setting differs in data, script, adversarial drift, privacy, and device limits."),
    ("Boundary class", "— smishing sits lexically between legitimate transactional messages and bulk spam, making tri-class detection hard."),
])
foot(s)

# ============ 5. LITERATURE SURVEY intro ============
s = slide(); header(s, "Literature Survey")
txt(s, 0.6, 1.3, 12.1, 0.6, [[R("The review integrates 169 records across four evidence areas; 48 are direct SMS detectors "
    "(36 English-only, 11 non-English, 1 unstated).", 14.5, INK, False, BODY)]])
cards = [("D1 · SMS & Smishing Detection", "rule-based → ML → deep learning → PLM → hybrid content+URL"),
         ("D2 · Code-Mixed Indic NLP", "transliteration, language-ID, MuRIL / IndicBERT"),
         ("D3 · LLM Text-Threat Detection", "prompting, augmentation, fine-tuned small LMs"),
         ("D4 · Resource-Efficient On-Device ML", "quantization, PEFT, edge profiling")]
for i, (t, d) in enumerate(cards):
    x = 0.6 + (i % 2) * 6.15; y = 2.2 + (i // 2) * 1.5
    box(s, x, y, 5.9, 1.3, fill=LIGHT, line=LINE, rounded=True)
    box(s, x, y, 0.11, 1.3, fill=BLUE)
    txt(s, x + 0.3, y + 0.16, 5.5, 0.5, [[R(t, 14.5, INK, True, HEAD)]])
    txt(s, x + 0.3, y + 0.62, 5.5, 0.6, [[R(d, 12, MUTED, False, BODY)]])
txt(s, 0.6, 5.5, 12.1, 0.7, [[R("Evolution:  ", 13, BLUE, True, HEAD),
    R("2006 content filtering → 2011 SMS Spam Collection → 2017–19 BERT era → 2020 CNN-LSTM hybrids → "
      "2021 MuRIL/Dravidian corpora → 2022 tri-class dataset → 2023 QLoRA/LLM detection → 2025–26 few-shot SLMs & the open Indic gap.",
      12.5, INK, False, BODY)]])
foot(s)

# ============ 6-7. LITERATURE SURVEY TABLE ============
lit = [
    ("Almeida et al. (2011)", "Released the SMS Spam Collection benchmark + classifier baselines", "Near-duplicate spam; English-only; now saturated"),
    ("Gómez-Hidalgo et al. (2006)", "First bag-of-words SMS spam filtering; feasibility shown", "Pre-benchmark, small operator corpora"),
    ("Roy et al. (2020)", "CNN/LSTM beat tuned classical baselines on SMS spam", "Benchmark ceiling; English-only"),
    ("Ghourabi et al. (2020)", "CNN-LSTM detects Arabic+English SMS spam", "Only two languages; no code-mix"),
    ("Mishra & Soni (2020)", "Smishing Detector: content+URL, 96.2% accuracy", "Blind to link-free / callback lures"),
    ("Uddin et al. (2025)", "ExplainableDetector: RoBERTa, 99.84% + LIME/attribution", "Saturated 2011 English benchmark"),
    ("Tanbhir et al. (2024)", "BERT + char-CNN Bangla smishing, 98.47%", "Single language; small corpus"),
    ("Mambina et al. (2022)", "TF-IDF Swahili smishing, 99.86% (low-resource)", "Single-country, single-language corpus"),
    ("Salman et al. (2024)", "67K-msg corpus; evasion stress-testing of filters", "Binary labels; shallow-model fragility"),
    ("Lee & Han (2024)", "KorSmishing Explainer: Korean LLM +15% over GPT-4 + rationales", "Korean-only; no edge / robustness"),
    ("Salman et al. (2025)", "SpaLLM-Guard: LLMs on SMS under manipulation & drift", "No on-device profiling; English"),
    ("COPS (2024)", "On-device pipeline; time+memory on Galaxy S21 FE", "No energy / quantized precision; English"),
    ("Krawczyk et al. (2026)", "XLM-R + LoRA, 97% F1 from 50–250 samples/lang", "No quantization; no Indic; no on-device"),
    ("Ramanujam & Abirami (2023)", "First Dravidian SMS corpus (7,700) incl. Roman-script Tamil", "Spam/ham only; no smishing class"),
    ("Seo et al. (2024)", "On-device char-CNN, evasion-resistant handset prototype", "Incomplete resource profile; no bit-width"),
    ("Khanuja/Kakwani (MuRIL, IndicBERT)", "Transliteration-aware Indic encoders; strong baselines", "Not evaluated on fraud-labelled Indian SMS"),
]
for part in range(2):
    s = slide(); header(s, "Literature Survey", contd=True)
    chunk = lit[:8] if part == 0 else lit[8:]
    rows = [("Sl. No.", "Article", "Strength", "Limitations")]
    for k, (a, st, li) in enumerate(chunk):
        rows.append((str((0 if part == 0 else 8) + k + 1), a, st, li))
    table(s, 0.45, 1.35, 12.45, [0.7, 3.0, 5.0, 3.75], rows, fs=10, hfs=11, rh=0.66)
    if part == 1:
        txt(s, 0.45, 6.75, 12.4, 0.4, [[R("Table 1: Summary of representative literature (16 articles).", 10, MUTED, True, BODY)]])
    foot(s)

# ============ 8. RESEARCH GAP (2-col) ============
s = slide(); header(s, "Research Gap")
bullets(s, 0.55, 1.35, 6.5, [
    ("Four mature literatures, no intersection", "— each has been reviewed in isolation; none integrates all four for the target task."),
    ("No code-mixed Indian smishing corpus", "that separates ham, bulk spam, and smishing — tri-class data exist only in English and single-script Bangla."),
    ("Indic encoders untested on the task", "— no MuRIL/IndicBERT fine-tuning on fraud-labelled Indian SMS in the reviewed evidence."),
    ("No joint deployment evaluation", "— no detector reports accuracy + robustness + latency + memory + energy at a stated on-device precision together."),
    ("Evaluation is leakage-prone", "— near-duplicate contamination inflates reported scores across the field."),
], size=13.5, gap=11)
# right-side diagram: four boxes -> gap
gx = 7.5
labels = ["D1  SMS detection", "D2  Code-mixed Indic NLP", "D3  LLM threat detection", "D4  On-device efficiency"]
for i, l in enumerate(labels):
    box(s, gx + (i % 2) * 2.65, 1.5 + (i // 2) * 0.95, 2.5, 0.8, fill=LIGHT, line=LINE, rounded=True)
    txt(s, gx + (i % 2) * 2.65, 1.5 + (i // 2) * 0.95, 2.5, 0.8, [[R(l, 11.5, INK, True, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
txt(s, gx, 3.5, 5.15, 0.5, [[R("↓   integrated for the first time   ↓", 12, BLUE, True, BODY)]], align=PP_ALIGN.CENTER)
box(s, gx + 0.4, 4.1, 4.35, 1.9, fill=NAVY, rounded=True)
txt(s, gx + 0.65, 4.3, 3.85, 1.5, [
    [R("THE GAP", 12, RGBColor(0x8F,0xAB,0xFF), True, HEAD)],
    [R("Tri-class, dual-script Malayalam–English smishing detection with an Indic encoder, evaluated on-device.", 12.5, WHITE, True, BODY)],
], anchor=MSO_ANCHOR.MIDDLE, sp=6)
txt(s, gx, 6.15, 5.15, 0.4, [[R("Fig. 1: Four evidence areas and the unaddressed intersection.", 9.5, MUTED, True, BODY)]], align=PP_ALIGN.CENTER)
foot(s)

# ============ 9. PROBLEM STATEMENT ============
s = slide(); header(s, "Problem Statement")
box(s, 1.2, 2.0, 10.9, 2.5, fill=LIGHT, line=BLUE, lw=1.4, rounded=True)
txt(s, 1.7, 2.35, 9.9, 1.9, [
    [R("Design and deploy a ", 19, INK, False, BODY),
     R("tri-class (ham / spam / smishing) detector for code-mixed ", 19, INK, True, BODY),
     R("Malayalam–English SMS", 19, BLUE_DK, True, BODY),
     R(" that ", 19, INK, False, BODY),
     R("runs entirely on-device", 19, BLUE_DK, True, BODY),
     R(" — preserving user privacy under handset resource limits — and evaluate it under a "
       "leakage-resistant, script-controlled protocol.", 19, INK, False, BODY)],
], anchor=MSO_ANCHOR.MIDDLE, ls=1.25)
for i, t in enumerate(["No message leaves the device", "Full-precision accuracy", "Free & reproducible"]):
    x = 1.2 + i * 3.75
    box(s, x, 4.9, 3.5, 0.8, fill=WHITE, line=LINE, rounded=True)
    txt(s, x, 4.9, 3.5, 0.8, [[R(t, 13, BLUE_DK, True, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
foot(s)

# ============ 10. OBJECTIVES ============
s = slide(); header(s, "Objectives")
objs = [
    ("Build the corpus", "Assemble a governed tri-class, dual-script Malayalam–English SMS corpus using open datasets and persuasion-conditioned augmentation."),
    ("Establish baselines", "Fine-tune and compare classical, MuRIL, and IndicBERT detectors under a frozen, leakage-resistant protocol."),
    ("Study compression & script", "Measure how int8 quantization and Roman→native transliteration affect the security-critical smishing class."),
    ("Deploy on-device", "Ship the detector to run fully in-browser (ONNX + WebAssembly) — no server, message never leaves the device."),
]
for i, (t, d) in enumerate(objs):
    y = 1.5 + i * 1.25
    box(s, 0.6, y, 0.85, 0.95, fill=BLUE, rounded=True)
    txt(s, 0.6, y, 0.85, 0.95, [[R(f"O{i+1}", 20, WHITE, True, HEAD)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 1.65, y, 11.05, 0.95, fill=LIGHT, line=LINE, rounded=True)
    txt(s, 1.95, y + 0.13, 10.5, 0.7, [[R(t + ".  ", 15, INK, True, HEAD), R(d, 13, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
foot(s)

# ============ 11. METHODOLOGY 1 — pipeline flowchart ============
s = slide(); header(s, "Proposed Methodology")
txt(s, 0.6, 1.25, 12.1, 0.4, [[R("On-device detection pipeline (dependency-aware reference architecture).", 14, INK, True, HEAD)]])
steps = [("Incoming SMS", "native / Roman / code-mixed"), ("Script & Language ID", "token-level tags"),
         ("Normalization", "raw + optional IndicXlit"), ("Detector", "IndicBERT (ONNX) @ fp32"),
         ("Decision", "ham / spam / smishing"), ("User action", "allow / warn / quarantine")]
n = len(steps); w = 1.92; gap = 0.13; x0 = (13.333 - (n * w + (n - 1) * gap)) / 2
for i, (t, d) in enumerate(steps):
    x = x0 + i * (w + gap)
    fbox(s, x, 2.05, w, 1.15, t, d, fill=(NAVY if i in (3,) else LIGHT), tcol=(WHITE if i == 3 else INK), line=(NAVY if i == 3 else LINE))
    if i < n - 1: arrow(s, x + w - 0.02, 2.05, gap + 0.15, 1.15)
# supporting branches
box(s, 1.6, 3.75, 3.2, 0.85, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, rounded=True)
txt(s, 1.6, 3.75, 3.2, 0.85, [[R("Governed corpus", 12, BLUE_DK, True, BODY)], [R("deduplicated group/time splits", 10, MUTED, False, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, sp=1)
box(s, 5.1, 3.75, 3.2, 0.85, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, rounded=True)
txt(s, 5.1, 3.75, 3.2, 0.85, [[R("Latency / memory / energy", 12, BLUE_DK, True, BODY)], [R("robustness harness", 10, MUTED, False, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, sp=1)
box(s, 8.6, 3.75, 3.2, 0.85, fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE, rounded=True)
txt(s, 8.6, 3.75, 3.2, 0.85, [[R("Optional federated update", 12, BLUE_DK, True, BODY)], [R("privacy + poisoning controls", 10, MUTED, False, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, sp=1)
box(s, 0.6, 5.05, 12.1, 0.9, fill=LIGHT, line=LINE, rounded=True)
txt(s, 0.9, 5.18, 11.5, 0.65, [[R("Local classification path — any networked lookup must be disclosed and evaluated separately.  ", 12.5, INK, True, BODY),
    R("Script/language ID precedes optional transliteration; profiling and robustness apply at the deployed precision.", 12, MUTED, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
txt(s, 0.6, 6.05, 12.1, 0.35, [[R("Fig. 2: End-to-end methodology.", 9.5, MUTED, True, BODY)]])
foot(s)

# ============ 12. METHODOLOGY 2 — corpus construction ============
s = slide(); header(s, "Proposed Methodology", contd=True)
txt(s, 0.6, 1.25, 12.1, 0.4, [[R("Corpus construction — grounding real register, real fraud, and theory-guided synthesis.", 14, INK, True, HEAD)]])
src = [("DravidianCodeMix", "~20K Ml–En comments → real code-mixed ham"),
       ("Mishra & Soni", "5,971 tri-class (638 smishing) → seeds"),
       ("UCI SMS Spam", "5,574 English ham / spam → volume")]
for i, (t, d) in enumerate(src):
    fbox(s, 0.6, 2.0 + i * 1.0, 3.3, 0.9, t, d)
    arrow(s, 3.9, 2.0 + i * 1.0, 0.55, 0.9)
box(s, 4.5, 2.3, 2.4, 2.3, fill=NAVY, rounded=True)
txt(s, 4.5, 2.3, 2.4, 2.3, [[R("Fuse +", 14, WHITE, True, HEAD)], [R("balance", 14, WHITE, True, HEAD)], [R("(cap ham 3,000)", 10.5, RGBColor(0xB9,0xC6,0xE6), False, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, sp=3)
arrow(s, 6.9, 2.3, 0.55, 2.3)
fbox(s, 7.5, 2.0, 5.2, 1.35, "Persuasion-conditioned augmentation",
     "Llama-3.3-70B (Groq): Cialdini principles (authority, scarcity, urgency…); link-based + link-free scam patterns; 573 reviewed", fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE)
fbox(s, 7.5, 3.5, 5.2, 1.1, "Dedup → stratified split",
     "exact + near-duplicate removal BEFORE splitting; 70/15/15; synthetic is train-only", fill=RGBColor(0xEF,0xF3,0xFF), line=BLUE)
box(s, 0.6, 5.05, 12.1, 1.05, fill=LIGHT, line=LINE, rounded=True)
txt(s, 0.9, 5.2, 11.5, 0.8, [[R("Final corpus:  ", 13, BLUE, True, HEAD),
    R("5,174 messages — 3,034 ham · 1,399 smishing · 741 spam.  Splits: train 3,937 · val 618 · test 619.  "
      "Ham hard-negatives (real OTPs, bank alerts, deliveries) added so that “numbers + an action” alone are not read as fraud.",
      12.5, INK, False, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
foot(s)

# ============ 13. METHODOLOGY 3 — models & protocol ============
s = slide(); header(s, "Proposed Methodology", contd=True)
txt(s, 0.6, 1.25, 6.2, 0.4, [[R("Models under test", 14, INK, True, HEAD)]])
bullets(s, 0.6, 1.75, 6.3, [
    ("Classical control", "TF-IDF (char n-grams) + Random Forest, class-balanced."),
    ("MuRIL", "16-language Indic encoder, transliteration-aware."),
    ("IndicBERT", "compact ALBERT-style, 200K SentencePiece vocab, ~135 MB."),
    ("Class-weighted loss", "to keep the minority smishing class from collapsing into spam."),
    ("Compression", "IndicBERT → ONNX; fp32 vs int8 studied per-class."),
], size=13, gap=9)
txt(s, 7.15, 1.25, 5.5, 0.4, [[R("Evaluation protocol (frozen)", 14, INK, True, HEAD)]])
bullets(s, 7.15, 1.75, 5.5, [
    ("Macro-F1 primary", "+ per-class smishing recall (minority class)."),
    ("Dedup before split", "+ group/near-duplicate grouping to stop leakage."),
    ("Script as a variable", "Roman / native / transliterated arms."),
    ("Calibration & FP rate", "ECE + false positives per 1,000 ham."),
    ("Frozen before training", "— honest, reproducible, not tuned to the test set."),
], size=13, gap=9)
foot(s)

# ============ 14. RESULTS 1 — protocol validation (leakage) ============
s = slide(); header(s, "Results & Discussion")
txt(s, 0.6, 1.25, 12.1, 0.55, [[R("Part A — Protocol validation on proxy corpora.  ", 15, BLUE, True, HEAD),
    R("Deduplication + near-duplicate grouping quantify how leakage inflates scores.", 13.5, INK, False, BODY)]])
rows = [("Model", "Split", "Macro-F1", "Smishing recall", "Clusters crossing split"),
        ("Word TF-IDF SVM", "Naïve", "0.916", "0.914", "50–64"),
        ("Word TF-IDF SVM", "Grouped", "0.885", "0.887", "0"),
        ("Char 2–5-gram SVM", "Naïve", "0.921", "0.919", "50–64"),
        ("Char 2–5-gram SVM", "Grouped", "0.894", "0.897", "0")]
table(s, 0.7, 1.95, 11.9, [3.0, 1.9, 2.0, 2.5, 2.5], rows, fs=12, hfs=12, rh=0.5)
box(s, 0.7, 4.85, 11.9, 1.35, fill=LIGHT, line=LINE, rounded=True)
txt(s, 1.0, 5.0, 11.3, 1.05, [
    [R("Grouping near-duplicates lowers mean macro-F1 by 0.031 (word) and 0.027 (char) vs naïve splitting.", 13.5, INK, True, BODY)],
    [R("Naïve splits let 50–64 near-duplicate clusters cross train/test; grouped splits allow none. "
       "Leakage — not architecture — explains much of the field's headline scores.", 12.5, MUTED, False, BODY)],
], anchor=MSO_ANCHOR.MIDDLE, sp=6)
txt(s, 0.7, 6.25, 11.9, 0.35, [[R("Table 2: Five-seed tri-class results, SMS Phishing Dataset (naïve vs near-duplicate-grouped).", 9.5, MUTED, True, BODY)]])
foot(s)

# ============ 15. RESULTS 2 — Bangla + UCI contamination ============
s = slide(); header(s, "Results & Discussion", contd=True)
txt(s, 0.6, 1.25, 12.1, 0.5, [[R("Part A — cross-corpus audit: duplication & contamination are pervasive.", 15, BLUE, True, HEAD)]])
# left: dedup stats
box(s, 0.6, 1.95, 5.9, 4.1, fill=LIGHT, line=LINE, rounded=True)
txt(s, 0.9, 2.15, 5.3, 0.4, [[R("EXACT DEDUPLICATION", 11, BLUE, True, HEAD)]])
for i, (a, b, c) in enumerate([("English (SMS Phishing)", "5,971", "5,797"),
                                ("Bangla (BangalaBarta)", "2,772", "1,204")]):
    y = 2.65 + i * 1.1
    txt(s, 0.9, y, 5.3, 0.4, [[R(a, 12.5, INK, True, BODY)]])
    txt(s, 0.9, y + 0.38, 5.3, 0.55, [[R(b + " rows", 15, MUTED, True, HEAD), R("  →  ", 13, INK, False, BODY), R(c + " independent", 15, BLUE_DK, True, HEAD)]])
txt(s, 0.9, 5.0, 5.3, 0.9, [[R("Row count is not corpus diversity — ", 12, INK, True, BODY),
    R("BangalaBarta loses 57% of rows to exact duplicates.", 12, MUTED, False, BODY)]])
# right: UCI contamination
box(s, 6.7, 1.95, 5.95, 4.1, fill=NAVY, rounded=True)
txt(s, 7.0, 2.15, 5.4, 0.4, [[R("UCI TRANSFER-TEST PROVENANCE AUDIT", 11, RGBColor(0x8F,0xAB,0xFF), True, HEAD)]])
txt(s, 7.0, 2.75, 5.4, 1.4, [
    [R("4,751 / 5,159", 30, WHITE, True, HEAD)],
    [R("unique UCI messages overlap the English source", 12.5, RGBColor(0xC7,0xD2,0xEC), False, BODY)],
], sp=4)
box(s, 7.0, 4.25, 5.35, 0.02, fill=RGBColor(0x35,0x4B,0x82))
txt(s, 7.0, 4.4, 5.4, 1.4, [
    [R("Only 408", 24, RGBColor(0x8F,0xD6,0xB0), True, HEAD), R("  contamination-clean cases remain.", 13, WHITE, True, BODY)],
    [R("An unfiltered transfer score would be a contaminated evaluation, not evidence of generalization.", 12, RGBColor(0xC7,0xD2,0xEC), False, BODY)],
], sp=6)
foot(s)

# ============ 16. RESULTS 3 — IndoSmish benchmark ============
s = slide(); header(s, "Results & Discussion", contd=True)
txt(s, 0.6, 1.25, 12.1, 0.55, [[R("Part B — IndoSmish implementation: the code-mixed detector we built.  ", 15, BLUE, True, HEAD),
    R("Held-out test, Roman arm.", 13, INK, False, BODY)]])
rows = [("Model", "Type", "Macro-F1", "Smishing recall"),
        ("TF-IDF + Random Forest", "Classical", "0.807", "0.645"),
        ("MuRIL", "Indic encoder", "0.805", "0.803"),
        ("IndicBERT   ● deployed", "Indic encoder", "0.892", "0.868")]
table(s, 1.4, 2.0, 10.5, [4.3, 2.7, 1.8, 1.7], rows, fs=13, hfs=12.5, rh=0.6)
# highlight best row visually
txt(s, 1.4, 4.75, 10.5, 0.8, [[R("IndicBERT wins on every metric and is the smallest model", 15, BLUE_DK, True, HEAD),
    R(" — the most accurate encoder is also the most deployable.", 14, INK, False, BODY)]], align=PP_ALIGN.CENTER)
# mini bars
for i, (lbl, val, col) in enumerate([("Classical", 0.645, MUTED), ("MuRIL", 0.803, BLUE), ("IndicBERT", 0.868, GREEN)]):
    x = 2.4 + i * 3.2
    txt(s, x, 5.55, 2.6, 0.3, [[R(lbl + " smishing recall", 10.5, MUTED, True, BODY)]], align=PP_ALIGN.CENTER)
    box(s, x, 5.9, 2.6, 0.34, fill=RGBColor(0xEE,0xF1,0xF7))
    box(s, x, 5.9, 2.6 * val, 0.34, fill=col)
    txt(s, x, 5.9, 2.6, 0.34, [[R(f"{val:.3f}", 11, WHITE, True, BODY)]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
foot(s)

# ============ 17. RESULTS 4 — three findings ============
s = slide(); header(s, "Results & Discussion", contd=True)
txt(s, 0.6, 1.25, 12.1, 0.45, [[R("Part B — three findings mapping to the paper's gaps.", 15, BLUE, True, HEAD)]])
finds = [("AUGMENTATION (G1)", "Code-mixed data unlocks the boundary class", "0.00", "0.803", GREEN,
          "MuRIL smishing recall before → after augmentation. Without it, smishing collapses into spam."),
         ("QUANTIZATION (G5)", "int8 disproportionately harms fraud detection", "0.868", "0.368", RED,
          "Dynamic int8 halves smishing recall (all configs) while ham stays perfect → ship at fp32."),
         ("SCRIPT (G2)", "A measurable transliteration deficit", "0.868", "0.658", RED,
          "Same messages, Roman → native script: −21 points smishing recall (Krishnan et al. confirmed).")]
for i, (tag, title, a, b, col, desc) in enumerate(finds):
    x = 0.6 + i * 4.1
    box(s, x, 1.9, 3.9, 4.3, fill=WHITE, line=LINE, rounded=True)
    box(s, x, 1.9, 3.9, 0.09, fill=col)
    txt(s, x + 0.25, 2.15, 3.4, 0.35, [[R(tag, 10.5, BLUE, True, HEAD)]])
    txt(s, x + 0.25, 2.55, 3.4, 1.0, [[R(title, 14, INK, True, HEAD)]])
    txt(s, x + 0.25, 3.75, 3.4, 0.9, [[R(a, 22, MUTED, True, HEAD), R("  →  ", 15, INK, False, BODY), R(b, 30, col, True, HEAD)]], anchor=MSO_ANCHOR.MIDDLE)
    txt(s, x + 0.25, 4.9, 3.4, 1.2, [[R(desc, 11, MUTED, False, BODY)]])
foot(s)

# ============ 18. RESULTS 5 — deployment ============
s = slide(); header(s, "Results & Discussion", contd=True)
txt(s, 0.6, 1.25, 12.1, 0.5, [[R("Part B — deployment: full-accuracy detection, on-device, free.", 15, BLUE, True, HEAD)]])
bullets(s, 0.6, 1.95, 6.7, [
    ("IndicBERT → ONNX → transformers.js", "runs on WebAssembly, entirely in the browser."),
    ("No server, no cold start", "— free to host and run; the SMS never leaves the device."),
    ("~15 ms per message on CPU", "— on-device-grade latency."),
    ("Live", "on Vercel + GitHub Pages as a professional web app."),
    ("Hardened iteratively", "— link-free scam patterns + ham hard-negatives added after real-world testing."),
], size=13.5, gap=10)
box(s, 7.6, 1.95, 5.1, 4.1, fill=LIGHT, line=LINE, rounded=True)
txt(s, 7.9, 2.15, 4.5, 0.4, [[R("LIVE VERDICTS (on-device)", 11, BLUE, True, HEAD)]])
for i, (msg, verd, col) in enumerate([('"reaching home by 8, dinner undo?"', "HAM", GREEN),
                                        ('"ningalude SBI account block aayi…"', "SMISHING", RED),
                                        ('"FLAT 50% OFF this weekend only!"', "SPAM", AMBER),
                                        ('"lost my phone, send 3000 to this no."', "SMISHING", RED)]):
    y = 2.7 + i * 0.82
    box(s, 7.9, y, 4.5, 0.68, fill=WHITE, line=LINE, rounded=True)
    txt(s, 8.05, y, 3.1, 0.68, [[R(msg, 10, INK, False, "Consolas")]], anchor=MSO_ANCHOR.MIDDLE)
    txt(s, 11.05, y, 1.25, 0.68, [[R(verd, 10.5, col, True, BODY)]], anchor=MSO_ANCHOR.MIDDLE)
foot(s)

# ============ 19. CONCLUSION ============
s = slide(white=False)
box(s, 0, 0, 13.333, 7.5, fill=NAVY)
box(s, 0, 0, 13.333, 0.16, fill=BLUE)
txt(s, 0.9, 0.7, 11, 0.9, [[R("Conclusion & Future Work", 30, WHITE, True, HEAD)]])
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.85), Inches(11.6), Inches(3.4))
tf = tb.text_frame; tf.word_wrap = True
concl = [
    ("A critical review", "integrates 169 records and 48 direct SMS detectors, exposing an empty four-literature intersection."),
    ("Protocol validated", "— dedup + grouping quantify leakage (−0.031 macro-F1); UCI is 92% contaminated."),
    ("IndoSmish built", "— a tri-class code-mixed detector: IndicBERT 0.892 macro-F1, deployed on-device."),
    ("Three findings", "— augmentation (G1), quantization (G5), and transliteration deficit (G2)."),
    ("Future work", "— quantized SLM (Qwen2.5 QLoRA), a native-script corpus, and named-handset energy profiling."),
]
for i, (h, d) in enumerate(concl):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(13); p.line_spacing = 1.1
    r = p.add_run(); r.text = "▪  "; r.font.color.rgb = BLUE; r.font.size = Pt(15); r.font.name = BODY
    r = p.add_run(); r.text = h + "  "; r.font.bold = True; r.font.color.rgb = WHITE; r.font.size = Pt(15.5); r.font.name = BODY
    r = p.add_run(); r.text = d; r.font.color.rgb = RGBColor(0xC7,0xD2,0xEC); r.font.size = Pt(14.5); r.font.name = BODY
box(s, 0.9, 5.5, 11.5, 0.02, fill=RGBColor(0x35,0x4B,0x82))
txt(s, 0.9, 5.7, 11.5, 1.0, [
    [R("Live demo:  ", 15, RGBColor(0x8F,0xAB,0xFF), True, BODY), R("indosmish-cid.vercel.app", 15, WHITE, True, BODY)],
    [R("Thank you.", 22, WHITE, True, HEAD)],
], sp=8)

out = ROOT / "IndoSmish_Colloquium.pptx"
prs.save(str(out))
print(f"Saved {len(prs.slides._sldIdLst)} slides -> {out}")
