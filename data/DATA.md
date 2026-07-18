# Source Datasets — Manual Download Steps

Place files in `data/raw/` with the names below, then update column mappings in
`configs/default.yaml` if the actual headers differ.

## 1. Dravidian Spam SMS (ham + spam source) — `dravidian_spam.csv`

- Ramanujam & Abirami, "Spam SMS in Dravidian languages," IEEE DataPort, 2023.
  DOI: 10.21227/dcym-pd69 → https://ieee-dataport.org/ (requires free IEEE account)
- ~7,700 messages, Tamil/Telugu/Kannada/Malayalam (incl. Roman-script Tamil),
  labels: ham/spam. We use the **Malayalam** subset (plus English-mixed rows).

## 2. SMS Phishing Dataset (smishing seed source) — `sms_phishing.csv`

- Mishra & Soni, Mendeley Data, DOI: 10.17632/f45bkkt8pr.1
  → https://data.mendeley.com/datasets/f45bkkt8pr/1 (direct download, no account)
- 5,971 messages: ham 4,844 / spam 489 / smishing 638 (English).
  Smishing rows are the augmentation seeds; a sample of ham/spam is also retained
  to keep an English fraction in the corpus (SMS code-mix includes pure-English turns).

## 3. Optional extras

- SmishTank (live smishing, English): https://smishtank.com — extra seeds.
- SMSDHL (TechRxiv, 2025): broader Indic spam coverage if Malayalam rows are thin.

## Provenance rules

- Record source, license, and access date here for each file you place.
- Raw files are **gitignored** — only the processed corpus and frozen splits are
  committed (with a dataset card).

| File | Source / DOI | License | Accessed | By |
|------|--------------|---------|----------|----|
|      |              |         |          |    |
