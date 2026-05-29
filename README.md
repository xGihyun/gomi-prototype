# Gomi

Technical debt risk analyzer for git repositories. Combines DistilBERT commit-message sentiment with Lizard static complexity and a Logistic Regression risk model trained on DeepJIT + ApacheJIT.

## How it works

| Layer | What                   | Input                                                                                |
| ----- | ---------------------- | ------------------------------------------------------------------------------------ |
| 1     | DistilBERT sentiment   | Commit messages → soft P(frustration)+P(caution) per commit                                          |
| 2     | Lizard static analysis | Source files → AvgCCN, AvgNLOC, function count (display only)                                        |
| 3     | Logistic Regression    | `[sentiment_score, low_info_ratio, change_entropy, satd_density, lt_normalized]` → risk probability  |
| 4     | SHAP LinearExplainer   | Decomposes risk score into per-feature contributions                                                  |

Analysis window: most recent 6 months of the repo's own commit history (anchored to latest commit, not wall clock). Files with fewer than 10 commits are flagged low-confidence.

---

## Setup

### 1. Install dependencies

```sh
uv sync
```

Optional deps unlock additional features:

```sh
# LoRA fine-tuning (reduces overfitting on small labeled set)
pip install peft

# Data augmentation (contextual word substitution)
pip install nlpaug

# Class imbalance resampling
pip install imbalanced-learn
```

### 2. Place datasets

All datasets go in `scripts/datasets/`.

| File                                        | Source                                                        | Used for                 |
| ------------------------------------------- | ------------------------------------------------------------- | ------------------------ |
| `openreview_labeled_2k.csv`                 | [OpenReview 2025](https://openreview.net/forum?id=FPLNSx1jmL) | DistilBERT fine-tuning   |
| `qt_test_raw.pkl` … `platform_test_raw.pkl` | [DeepJIT (ISSTA21)](https://github.com/ZZR0/ISSTA21-JIT-DP)   | LR training              |
| `apachejit_commits.csv`                     | [Zenodo 5907002](https://zenodo.org/records/5907002)          | LR training + validation |

DeepJIT pkl files: `qt`, `openstack`, `go`, `jdt`, `gerrit`, `platform` — each as `<name>_test_raw.pkl`. Feature files (`<name>_k_feature.csv`) are optional but add `change_entropy` and `lt_normalized` training features.

Required CSV columns:

- `openreview_labeled_2k.csv`: `message`, `reconciled_emotion`
- `apachejit_commits.csv`: `commit_hash`, `message`, `buggy`

---

## Training pipeline

Run once, in order. Each step is a prerequisite for the next.

### Step 1 — Vocabulary adaptation via FVT (optional but recommended)

Identifies high-frequency commit tokens fragmented by DistilBERT's WordPiece tokenizer, adds them to the vocabulary, and initializes their embeddings as the mean of their subword fragment embeddings (Fast Vocabulary Transfer). Saves to `scripts/datasets/distilbert_adapted/`. `train_sentiment.py` auto-detects this directory.

```sh
uv run python scripts/adapt_vocab.py
```

Runs in minutes on CPU. No GPU required.

### Step 2 — Fine-tune DistilBERT sentiment model

Fine-tunes on `openreview_labeled_2k.csv` with LoRA (if `peft` is installed) and data augmentation (if `nlpaug` is installed). Saves to `scripts/datasets/distilbert_sentiment/`.

```sh
uv run python scripts/train_sentiment.py
```

### Step 3 — Run Gomi

The LR model trains automatically on first run and caches to `scripts/datasets/risk_model.joblib`. Subsequent runs load from cache.

```sh
uv run python gomi.py <path-to-repo>
```

Show top N files (default 10):

```sh
uv run python gomi.py <path-to-repo> 20
```

---

## Output

```
  src/auth/middleware.py    [████████░░] 0.81  HIGH RISK ⚠
  src/db/migrations.py      [█████░░░░░] 0.52  MODERATE  ~
  src/utils/helpers.py      [██░░░░░░░░] 0.18  LOW       ✓
```

Drilldown for top 5 files includes:

- **Risk score** with SHAP breakdown per feature
- **Complexity**: AvgCCN, AvgNLOC, function count (from Lizard)
- **TODO/FIXME/HACK/XXX** comment counts
- **Git stats**: lines added/deleted, commit count, unique devs, age, change entropy
- **Commit sentiment**: per-message DistilBERT label

### SHAP breakdown

```
  base_rate          +0.3102
  sentiment_contrib  +0.2841   (sentiment_score = 0.667)
  low_info_contrib   -0.0412   (low_info_ratio = 0.200)
  entropy_contrib    +0.1203   (change_entropy = 0.812)
  satd_contrib       +0.0334   (satd_density = 0.043)
  complexity_contrib +0.0198   (lt_normalized / AvgNLOC = 0.731)
  ────────────────────────────
  ≈ risk score       +0.7266
```

---

## Retraining

Delete cached models to force a full retrain:

```sh
# Force LR retrain (fast — seconds)
rm scripts/datasets/risk_model.joblib

# Force DistilBERT retrain (slow — requires GPU)
rm -rf scripts/datasets/distilbert_sentiment/

# Force vocabulary re-adaptation
rm -rf scripts/datasets/distilbert_adapted/
```
