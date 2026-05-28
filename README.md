# Gomi

Technical debt risk analyzer for git repositories. Combines DistilBERT commit-message sentiment with Lizard static complexity and a Logistic Regression risk model trained on DeepJIT + ApacheJIT.

## How it works

| Layer | What                   | Input                                                                                |
| ----- | ---------------------- | ------------------------------------------------------------------------------------ |
| 1     | DistilBERT sentiment   | Commit messages → Frustration / Caution / Neutral / Satisfaction                     |
| 2     | Lizard static analysis | Source files → AvgCCN, AvgNLOC, function count (display only)                        |
| 3     | Logistic Regression    | `[sentiment_score, low_info_ratio, change_entropy, satd_density]` → risk probability |
| 4     | SHAP LinearExplainer   | Decomposes risk score into per-feature contributions                                 |

Analysis window: last 6 months of git history. Files with fewer than 10 commits are flagged low-confidence.

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

DeepJIT pkl files: `qt`, `openstack`, `go`, `jdt`, `gerrit`, `platform` — each as `<name>_test_raw.pkl`. Entropy feature files (`<name>_k_feature.csv`) are optional but improve the model.

Required CSV columns:

- `openreview_labeled_2k.csv`: `message`, `reconciled_emotion`
- `apachejit_commits.csv`: `commit_hash`, `message`, `buggy`

---

## Training pipeline

Run once, in order. Each step is a prerequisite for the next.

### Step 1 — MLM domain pre-training (optional but recommended)

Pre-trains DistilBERT on raw commit messages from DeepJIT + ApacheJIT before fine-tuning. Saves to `scripts/datasets/distilbert_commit_mlm/`. `train_sentiment.py` auto-detects this directory.

```sh
uv run python scripts/pretrain_mlm.py
```

~2000 training steps. Requires GPU for reasonable speed (~20 min on GPU, several hours on CPU).

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
  ────────────────────────────
  ≈ risk score       +0.7068
```

---

## Retraining

Delete cached models to force a full retrain:

```sh
# Force LR retrain (fast — seconds)
rm scripts/datasets/risk_model.joblib

# Force DistilBERT retrain (slow — requires GPU)
rm -rf scripts/datasets/distilbert_sentiment/

# Force MLM pre-training (slowest)
rm -rf scripts/datasets/distilbert_commit_mlm/
```
