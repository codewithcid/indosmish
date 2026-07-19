# Source Datasets — All Free (no IEEE / no paid access)

The gated IEEE DataPort Dravidian SMS corpus is **not required**. This project builds
the code-mixed tri-class corpus from free sources + persuasion-conditioned augmentation,
exactly as the paper's S1 recipe prescribes (real code-mixed register from DravidianCodeMix
[ref 11] + real smishing pretexts from Mishra & Soni [ref 48] + theory-guided augmentation
[ref 24]).

## Auto-downloaded (run this — no manual steps)

```powershell
python -m indosmish.data.fetch_free_data
```

This pulls, via the HuggingFace `datasets` library (all free, public):
- **DravidianCodeMix Malayalam–English** (`offenseval_dravidian`, config `malayalam`,
  ~20k real code-mixed comments) → `data/raw/dravidian_codemix_ml.csv`.
  `Not_offensive` rows become real code-mixed **ham** and augmentation style seeds.
- **UCI SMS Spam Collection** (`sms_spam`, English ham/spam) → `data/raw/uci_sms.csv`.

## Manual (1 click each, free, NO account)

### Mishra & Soni tri-class — `sms_phishing.csv`  [REQUIRED — smishing seeds]
- https://data.mendeley.com/datasets/f45bkkt8pr/1  → Download → unzip
- 5,971 msgs: ham 4,844 / spam 489 / smishing 638 (English).
- Save the message/label CSV as `data/raw/sms_phishing.csv`.
- If headers differ from `TEXT`/`LABEL`, fix them in `configs/default.yaml`.

### (Optional bonus) Balanced Spam/Smishing LLM set — `balanced_smishing.csv`
- https://data.mendeley.com/datasets/vmg875v4xs/1  → free download
- Extra smishing seeds; add as a source in the config if you want more augmentation diversity.

## How the three classes are assembled (all free)

| Class | Source |
|-------|--------|
| **ham** | English ham (Mishra&Soni, UCI) + real code-mixed ham (DravidianCodeMix `Not_offensive`) |
| **spam** | English spam (Mishra&Soni, UCI); a portion Gemini-transformed to code-mix for balance |
| **smishing** | English smishing seeds (Mishra&Soni) → Gemini persuasion-conditioned code-mix augmentation |

## Provenance log

| File | Source / DOI | License | Accessed | By |
|------|--------------|---------|----------|----|
| dravidian_codemix_ml.csv | DravidianCodeMix, arXiv:2106.09460 | CC BY 4.0 | | |
| uci_sms.csv | UCI SMS Spam Collection | CC BY 4.0 | | |
| sms_phishing.csv | Mendeley 10.17632/f45bkkt8pr.1 | CC BY 4.0 | | |
