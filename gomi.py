# gomi.py — Gomi CLI
# Pipeline: DistilBERT sentiment → Lizard complexity → LR risk fusion → SHAP
#
# Layer 1: DistilBERT fine-tuned on OpenReview 2025 (2,000 labeled commit messages)
#          Labels: Satisfaction | Frustration | Caution | Neutral
#          Sentiment Score per file = (Frustration + Caution) / total commits
#
# Layer 2: Lizard static analysis — AvgCCN, AvgNLOC, function_cnt, PARAM
#          Complexity Score = percentile rank of AvgCCN within this repo
#
# Layer 3: Logistic Regression trained on DeepJIT (6 projects) + ApacheJIT (14 projects)
#          Input: [Sentiment Score, Low-Info Ratio, Change Entropy, SATD Density]
#          Output: Risk Score (0–1 probability of being bug-prone)
#          SMOTE-Tomek resampling corrects class imbalance before fitting.
#          Structural complexity (Lizard) is a separate display metric — not an LR input
#          because it cannot be computed from training datasets (no source code).
#
# Layer 4: SHAP LinearExplainer — decomposes Risk Score into per-feature contributions
#
# Runtime window: 6 months of commit history per file
# Confidence threshold: files with < 10 commits are flagged as low confidence

import csv
import math
import os
import pickle
import re
import subprocess
import sys
import time

import joblib
import lizard
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from transformers import pipeline as hf_pipeline

# ─── PATHS ────────────────────────────────────────────────────────────────────

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/datasets")

# Layer 1 — DistilBERT fine-tuned on OpenReview 2025
# Expected: a saved HuggingFace model directory produced by fine-tuning
# distilbert-base-uncased on openreview_labeled_2k.csv.
# Train script: scripts/train_sentiment.py (not part of this runtime file).
SENTIMENT_MODEL_DIR = os.path.join(DATASET_DIR, "distilbert_sentiment")

# Fallback: raw OpenReview CSV used only to verify label mapping at startup.
OPENREVIEW_CSV = os.path.join(DATASET_DIR, "openreview_labeled_2k.csv")

# Layer 3 — DeepJIT raw pkl files (one per project)
# Source: https://github.com/ZZR0/ISSTA21-JIT-DP
DEEPJIT_PKLS = [
    os.path.join(DATASET_DIR, "qt_test_raw.pkl"),
    os.path.join(DATASET_DIR, "openstack_test_raw.pkl"),
    os.path.join(DATASET_DIR, "go_test_raw.pkl"),
    os.path.join(DATASET_DIR, "jdt_test_raw.pkl"),
    os.path.join(DATASET_DIR, "gerrit_test_raw.pkl"),
    os.path.join(DATASET_DIR, "platform_test_raw.pkl"),
]

# Layer 3 — ApacheJIT CSV (106,674 commits across 14 Apache projects)
# Source: https://zenodo.org/records/5907002
# Columns expected: commit_hash, message, buggy (0/1)
APACHEJIT_CSV = os.path.join(DATASET_DIR, "apachejit_commits.csv")

# Cached LR model — trained once, reused across runs.
# Delete this file to force retraining (e.g. after dataset changes).
LR_MODEL_CACHE = os.path.join(DATASET_DIR, "risk_model.joblib")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c",
    ".go", ".rs", ".rb", ".php",
}
SKIP_DIRS = {
    "node_modules", ".git", "vendor", "__pycache__",
    ".venv", "venv", "dist", "build", ".next",
}

# Ordered so FIXME matches before FIX (alternation is left-to-right)
TODO_PATTERN = re.compile(r"\b(TODO|FIXME|FIX|HACK|XXX)\b", re.IGNORECASE)

# Labels produced by the fine-tuned DistilBERT model
VALID_LABELS   = {"frustration", "caution", "neutral", "satisfaction"}
RISK_LABELS    = {"frustration", "caution"}

# Runtime analysis window: 6 months (as specified in thesis)
ANALYSIS_WINDOW_DAYS = 182

# Minimum commits before a file receives a risk score
MIN_COMMITS_FOR_SCORE = 10

# Step 1: Conventional Commit Prefix Stripping
# Regex matches: type(scope)!: description
CONVENTIONAL_COMMIT_REGEX = r"^[a-z]+(\([^)]+\))?!?:\s*"

# Step 2: Low-Information Message Handling
LOW_INFO_TOKEN_THRESHOLD = 5


# ─── LAYER 1: DistilBERT SENTIMENT MODEL ──────────────────────────────────────

def strip_prefix(message: str) -> str:
    """
    Strips conventional commit prefixes (e.g., 'fix(auth)!: ') from the message.
    Applied at both training and runtime for consistency.
    """
    if not message:
        return ""
    return re.sub(CONVENTIONAL_COMMIT_REGEX, "", message, flags=re.IGNORECASE).strip()


def is_low_info(message: str) -> bool:
    """
    Returns True if the message (after prefix stripping) has fewer than
    LOW_INFO_TOKEN_THRESHOLD tokens.
    """
    stripped = strip_prefix(message)
    tokens = stripped.split()
    return len(tokens) < LOW_INFO_TOKEN_THRESHOLD


def compute_msg_satd_density(message: str) -> float:
    """SATD keyword count / word count in message. 0.0 for empty messages."""
    tokens = message.split()
    if not tokens:
        return 0.0
    return len(TODO_PATTERN.findall(message)) / len(tokens)


def load_sentiment_model():
    """
    Loads the fine-tuned DistilBERT classifier from SENTIMENT_MODEL_DIR.

    The model must be pre-trained offline via scripts/train_sentiment.py,
    which fine-tunes distilbert-base-uncased on the OpenReview 2025 dataset
    (openreview_labeled_2k.csv) with labels: frustration, caution, neutral,
    satisfaction.

    Returns a HuggingFace text-classification pipeline ready for inference.
    The pipeline handles tokenization and softmax internally.
    """
    if not os.path.isdir(SENTIMENT_MODEL_DIR):
        print(
            f"\n  ERROR: Fine-tuned DistilBERT model not found at:\n"
            f"    {SENTIMENT_MODEL_DIR}\n\n"
            f"  Run scripts/train_sentiment.py first to fine-tune on OpenReview 2025.\n"
            f"  The script reads {OPENREVIEW_CSV} and saves the model to {SENTIMENT_MODEL_DIR}.\n"
        )
        sys.exit(1)

    classifier = hf_pipeline(
        "text-classification",
        model=SENTIMENT_MODEL_DIR,
        tokenizer=SENTIMENT_MODEL_DIR,
        top_k=None,          # return all class probabilities, not just argmax
        truncation=True,
        max_length=128,      # commit messages rarely exceed 128 tokens
    )
    print(f"  Loaded DistilBERT sentiment model from: {SENTIMENT_MODEL_DIR}")
    return classifier


def classify_message(message: str, classifier) -> str:
    """
    Classifies a single commit message using the fine-tuned DistilBERT model.

    Prefixes are stripped before classification for consistency with training.
    """
    stripped = strip_prefix(message)
    if not stripped:
        return "neutral"

    # pipeline returns list of lists when top_k=None: [[{label, score}, ...]]
    results = classifier(stripped[:512])[0]  # truncate safety; model also truncates
    best = max(results, key=lambda x: x["score"])
    label = best["label"].lower()

    # Normalize label to expected scheme in case model saved with capitalized labels
    label = label.replace("label_", "")  # handle HF auto-labeling e.g. "LABEL_0"
    if label not in VALID_LABELS:
        # Fall back: pick closest known label by string match
        for known in VALID_LABELS:
            if known in label:
                return known
        return "neutral"

    return label


def compute_sentiment_score(
    messages: list[str], classifier
) -> tuple[float, float, list[tuple[str, str]]]:
    """
    Processes a list of commit messages for a file.

    1. Identifies low-information messages (tokens < 5).
    2. low_info_ratio = low_info_count / total_commits
    3. Sentiment Score = (frustration + caution) / (total_commits - low_info_count)
       (If all commits are low-info, sentiment_score is 0.0)

    Returns (sentiment_score, low_info_ratio, [(message, label), ...]).
    """
    if not messages:
        return 0.0, 0.0, []

    total_count = len(messages)
    low_info_msgs = [m for m in messages if is_low_info(m)]
    low_info_ratio = len(low_info_msgs) / total_count

    # DistilBERT is only asked to classify non-low-info messages
    process_msgs = [m for m in messages if not is_low_info(m)]

    if not process_msgs:
        return 0.0, low_info_ratio, [(m, "low_info") for m in messages]

    labeled = [(msg, classify_message(msg, classifier)) for msg in process_msgs]
    risk_count = sum(1 for _, label in labeled if label in RISK_LABELS)
    sentiment_score = risk_count / len(process_msgs)

    # Combine for display/return
    final_labeled = []
    # Re-assemble labeled list to preserve original count/order if possible,
    # but here we just return them labeled.
    for m in messages:
        if is_low_info(m):
            final_labeled.append((m, "low_info"))
        else:
            # find its label from the labeled list
            label = "neutral"
            for msg, lbl in labeled:
                if msg == m:
                    label = lbl
                    break
            final_labeled.append((m, label))

    return sentiment_score, low_info_ratio, final_labeled



# ─── LAYER 2: LIZARD (Structural Complexity) ──────────────────────────────────

def compute_complexity(filepath: str) -> dict:
    """
    Runs Lizard on a single source file and extracts:
      - AvgCCN: average cyclomatic complexity across all functions
      - AvgNLOC: average lines of code per function
      - function_cnt: number of functions/methods detected
      - PARAM: average parameter count per function

    Returns zero-valued dict if the file has no parseable functions
    (e.g. pure data files, empty modules).
    """
    result = lizard.analyze_file(filepath)
    if not result.function_list:
        return {"AvgCCN": 0.0, "AvgNLOC": 0.0, "function_cnt": 0, "PARAM": 0.0}

    avg_ccn  = float(np.mean([f.cyclomatic_complexity for f in result.function_list]))
    avg_nloc = float(np.mean([f.nloc for f in result.function_list]))
    avg_param = float(np.mean([f.parameter_count for f in result.function_list]))
    return {
        "AvgCCN": avg_ccn,
        "AvgNLOC": avg_nloc,
        "function_cnt": len(result.function_list),
        "PARAM": avg_param,
    }


def count_todo_comments(filepath: str) -> dict:
    """
    Counts TODO/FIXME/FIX/HACK/XXX tags in source file.
    Display-only — not an LR input (no source code in training datasets).
    """
    tags = {"TODO": 0, "FIXME": 0, "FIX": 0, "HACK": 0, "XXX": 0}
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                for m in TODO_PATTERN.finditer(line):
                    tags[m.group(1).upper()] += 1
    except OSError:
        pass
    return tags


def percentile_rank(value: float, all_values: list[float]) -> float:
    """
    Returns the percentile rank of value within all_values, as a 0–1 float.
    A file with AvgCCN higher than 80% of files in the repo scores 0.80.

    This normalization is repo-relative, consistent with how the LR model
    was trained (each project normalized against its own baseline).
    """
    if not all_values or len(all_values) == 1:
        return 0.0
    return round(sum(1 for x in all_values if x <= value) / len(all_values), 4)


# ─── LAYER 3: LOGISTIC REGRESSION (Risk Fusion) ───────────────────────────────

def _load_deepjit_records(sentiment_clf) -> list[dict]:
    """
    Loads all available DeepJIT pkl files (up to 6 projects).
    Each pkl contains: [hashes, labels, messages, code_changes].

    For each project, the companion *_k_feature.csv is loaded if present —
    it contains per-commit entropy keyed by commit hash. Entropy is
    percentile-ranked within each project for repo-relative normalisation.

    Missing entropy (no CSV, or hash absent from CSV) is stored as np.nan.
    train_risk_model() imputes NaN values with the column mean over all
    records that have observed entropy (SimpleImputer, strategy='mean').

    Records from missing pkl files are skipped with a warning.
    """
    records = []
    for pkl_path in DEEPJIT_PKLS:
        if not os.path.isfile(pkl_path):
            project = os.path.basename(pkl_path).replace("_test_raw.pkl", "")
            print(f"    [skip] DeepJIT project '{project}' not found: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        hashes, labels, messages = raw[0], raw[1], raw[2]
        project = os.path.basename(pkl_path).replace("_test_raw.pkl", "")

        feature_csv = pkl_path.replace("_test_raw.pkl", "_k_feature.csv")
        ent_by_hash: dict[str, float] = {}
        if os.path.isfile(feature_csv):
            with open(feature_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        ent_by_hash[row["_id"]] = float(row["entrophy"])
                    except (ValueError, KeyError):
                        pass

        all_ent = list(ent_by_hash.values())
        has_entropy = bool(all_ent)
        # Project-mean percentile rank used to impute commits whose hash is absent
        # from the k_feature CSV (within-project mean imputation).
        project_mean_rank = (
            percentile_rank(float(np.mean(all_ent)), all_ent) if has_entropy else np.nan
        )
        print(f"    DeepJIT [{project}]: {len(labels)} commits  "
              f"(entropy: {'yes' if has_entropy else 'no — NaN, will impute'})")

        for h, msg, label in zip(hashes, messages, labels):
            low_info = 1.0 if is_low_info(msg) else 0.0

            if low_info:
                sentiment_score = 0.0
            else:
                emotion = classify_message(msg, sentiment_clf)
                sentiment_score = 1.0 if emotion in RISK_LABELS else 0.0

            if has_entropy and h in ent_by_hash:
                change_entropy = percentile_rank(ent_by_hash[h], all_ent)
            elif has_entropy:
                # Hash absent from CSV but project has entropy data: within-project mean.
                change_entropy = project_mean_rank
            else:
                change_entropy = np.nan  # imputed at training level

            records.append({
                "sentiment_score": sentiment_score,
                "low_info_ratio": low_info,
                "change_entropy": change_entropy,
                "satd_density": compute_msg_satd_density(msg),
                "buggy": int(label),
            })

    return records


def _load_apachejit_records(sentiment_clf) -> tuple[list[dict], list[dict]]:
    """
    Loads ApacheJIT commits from apachejit_commits.csv.
    Applies an 80/20 split: training records and held-out validation records.

    ApacheJIT ships no source code and no per-commit entropy file, so
    change_entropy is stored as np.nan and imputed by train_risk_model()
    using the mean of observed entropy values from DeepJIT records.

    Expected CSV columns: commit_hash, message, buggy
    """
    if not os.path.isfile(APACHEJIT_CSV):
        print(f"    [skip] ApacheJIT CSV not found: {APACHEJIT_CSV}")
        return [], []

    rows = []
    with open(APACHEJIT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                msg   = row.get("message", "").strip()
                buggy = int(row.get("buggy", 0))
                rows.append((msg, buggy))
            except (ValueError, KeyError):
                continue

    print(f"    ApacheJIT: {len(rows)} commits loaded")

    # Stratified-ish 80/20 split (deterministic)
    split_idx = int(len(rows) * 0.8)
    train_rows = rows[:split_idx]
    val_rows   = rows[split_idx:]

    def rows_to_records(row_list):
        recs = []
        for msg, buggy in row_list:
            low_info = 1.0 if is_low_info(msg) else 0.0

            if low_info:
                sentiment_score = 0.0
            else:
                emotion = classify_message(msg, sentiment_clf)
                sentiment_score = 1.0 if emotion in RISK_LABELS else 0.0

            recs.append({
                "sentiment_score": sentiment_score,
                "low_info_ratio": low_info,
                "change_entropy": np.nan,  # no entropy data in ApacheJIT; imputed at training level
                "satd_density": compute_msg_satd_density(msg),
                "buggy": buggy,
            })
        return recs

    print(f"    ApacheJIT split → {len(train_rows)} train / {len(val_rows)} validation")
    return rows_to_records(train_rows), rows_to_records(val_rows)



def train_risk_model(sentiment_clf) -> tuple:
    """
    Returns (trained LR model, X_train numpy array for SHAP background).

    On first run: trains on DeepJIT + ApacheJIT with SMOTE-Tomek resampling
    and saves the result to LR_MODEL_CACHE. Subsequent runs load from cache.
    Delete LR_MODEL_CACHE to force retraining.

    Feature vector: [sentiment_score, low_info_ratio, change_entropy]
    Missing change_entropy values (np.nan) are imputed with the column mean
    over observed values via sklearn SimpleImputer (standard mean imputation).
    """
    if os.path.isfile(LR_MODEL_CACHE):
        print(f"  Loading cached LR model from: {LR_MODEL_CACHE}")
        print("  (delete this file to force retraining)")
        cached = joblib.load(LR_MODEL_CACHE)
        return cached["model"], cached["X_train"]

    print("  Loading DeepJIT records...")
    deepjit_records = _load_deepjit_records(sentiment_clf)

    print("  Loading ApacheJIT records...")
    apache_train, apache_val = _load_apachejit_records(sentiment_clf)

    all_train = deepjit_records + apache_train

    if not all_train:
        print("\n  ERROR: No training data available. Check dataset paths.")
        sys.exit(1)

    X_train = np.array([
        [r["sentiment_score"], r["low_info_ratio"], r["change_entropy"], r["satd_density"]]
        for r in all_train
    ], dtype=float)
    y_train = np.array([r["buggy"] for r in all_train])

    # Mean imputation for change_entropy where data was unavailable (np.nan).
    # Imputation mean is computed from observed DeepJIT entropy values only —
    # no invented constant; the value is fully data-derived.
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    entropy_mean = imputer.statistics_[2]  # index 2 = change_entropy column
    nan_count = int(np.isnan(np.array([r["change_entropy"] for r in all_train], dtype=float)).sum())
    print(f"  change_entropy: {nan_count} NaN values imputed with observed mean {entropy_mean:.4f}")

    buggy_n = int(y_train.sum())
    print(f"  Raw training set: {len(y_train)} commits ({buggy_n} buggy, {len(y_train) - buggy_n} clean)")

    try:
        from imblearn.combine import SMOTETomek
        smt = SMOTETomek(random_state=42)
        X_train, y_train = smt.fit_resample(X_train, y_train)
        buggy_r = int(y_train.sum())
        print(f"  After SMOTE-Tomek: {len(y_train)} samples ({buggy_r} buggy, {len(y_train) - buggy_r} clean)")
    except ImportError:
        print("  [SMOTE-Tomek] imbalanced-learn not installed — skipping resampling")
        print("  Install with: pip install imbalanced-learn")

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    print(f"  LR coef → sentiment: {model.coef_[0][0]:.4f}  "
          f"low_info: {model.coef_[0][1]:.4f}  "
          f"entropy: {model.coef_[0][2]:.4f}  "
          f"satd: {model.coef_[0][3]:.4f}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")

    if apache_val:
        X_val = np.array([
            [r["sentiment_score"], r["low_info_ratio"], r["change_entropy"], r["satd_density"]]
            for r in apache_val
        ], dtype=float)
        X_val = imputer.transform(X_val)
        y_val = np.array([r["buggy"] for r in apache_val])
        preds = model.predict(X_val)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        print(f"  Validation (ApacheJIT held-out 20%): "
              f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    joblib.dump({"model": model, "X_train": X_train}, LR_MODEL_CACHE)
    print(f"  LR model cached → {LR_MODEL_CACHE}")

    return model, X_train


# ─── LAYER 4: SHAP ────────────────────────────────────────────────────────────

def compute_shap(model, X_train: np.ndarray, X_file: np.ndarray) -> dict:
    """
    Decomposes the Risk Score into per-feature contributions using SHAP.

    Uses LinearExplainer with X_train as the background distribution.
    For a four-feature LR model this is equivalent to:
      risk ≈ base_rate + sentiment_contrib + low_info_contrib + entropy_contrib + satd_contrib

    Returns contributions for the buggy class (class index 1).
    """
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_file)

    # shap_values may be a list [class0, class1] or a single array
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv   = shap_values[1][0]
        base = (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
    else:
        sv   = shap_values[0]
        base = (
            explainer.expected_value
            if not hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value[0]
        )

    return {
        "base_rate":         float(base),
        "sentiment_contrib": float(sv[0]),
        "low_info_contrib":  float(sv[1]),
        "entropy_contrib":   float(sv[2]),
        "satd_contrib":      float(sv[3]),
    }



# ─── GIT HISTORY (6-month window) ─────────────────────────────────────────────

def get_repo_git_stats(repo_path: str) -> dict:
    """
    Collects git history for the 6-month analysis window (ANALYSIS_WINDOW_DAYS).

    Per the thesis runtime spec:
      - Default commit window: 6 months per file
      - Files with < 10 commits are flagged as low confidence

    Handles subdirectory repos (e.g. myorg/client where .git is at myorg/):
    detects git root, computes path prefix, strips it from numstat output
    so file lookups match os.path.relpath(filepath, repo_path).

    Stats are DISPLAY ONLY — not fed into the risk model.
    Returns: {rel_filepath: {la, ld, nf, ndev, age_days, ent, messages}}
    """
    # Resolve true git root (may be a parent of repo_path)
    git_root = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    ).stdout.strip()

    rel_from_root = os.path.relpath(repo_path, git_root)
    path_prefix   = "" if rel_from_root == "." else rel_from_root.replace(os.sep, "/") + "/"

    since_flag = f"--since={ANALYSIS_WINDOW_DAYS} days ago"

    # Use ASCII record separator (0x1e) before each commit so blank lines
    # in numstat output never confuse the parser.
    proc = subprocess.run(
        ["git", "-C", repo_path, "log",
         "--format=%x1eCOMMIT\t%H\t%ae\t%at\t%s",
         "--numstat", "--diff-filter=AM",
         since_flag, "--", "."],
        capture_output=True, text=True, errors="replace",
    )

    raw: dict[str, list] = {}

    # Each chunk is one commit block (header line + numstat lines)
    for chunk in proc.stdout.split("\x1e"):
        lines = chunk.splitlines()
        if not lines:
            continue
        header = lines[0]
        if not header.startswith("COMMIT\t"):
            continue
        parts = header.split("\t", 4)
        current = {
            "author":  parts[2] if len(parts) > 2 else "",
            "ts":      int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
            "message": parts[4].strip() if len(parts) > 4 else "",
        }
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                try:
                    la       = int(parts[0]) if parts[0] not in ("-", "") else 0
                    ld       = int(parts[1]) if parts[1] not in ("-", "") else 0
                    filepath = parts[2].strip()
                    if path_prefix and filepath.startswith(path_prefix):
                        filepath = filepath[len(path_prefix):]
                    elif path_prefix:
                        continue
                    raw.setdefault(filepath, []).append({**current, "la": la, "ld": ld})
                except ValueError:
                    pass

    result = {}
    for filepath, changes in raw.items():
        total_la    = sum(c["la"] for c in changes)
        total_ld    = sum(c["ld"] for c in changes)
        nf          = len(changes)
        ndev        = len({c["author"] for c in changes})
        timestamps  = sorted(c["ts"] for c in changes)
        age_days    = (timestamps[-1] - timestamps[0]) / 86400.0 if len(timestamps) > 1 else 0.0
        total_churn = sum(c["la"] + c["ld"] for c in changes)
        if total_churn > 0:
            probs = [(c["la"] + c["ld"]) / total_churn for c in changes]
            ent   = -sum(p * math.log2(p) for p in probs if p > 0)
        else:
            ent = 0.0
        seen: set[str] = set()
        messages: list[str] = []
        for c in changes:
            if c["message"] and c["message"] not in seen:
                seen.add(c["message"])
                messages.append(c["message"])
        result[filepath] = {
            "la": total_la, "ld": total_ld, "nf": nf,
            "ndev": ndev, "age_days": round(age_days, 1),
            "ent": round(ent, 3), "messages": messages,
        }
    return result


def get_source_files(repo_path: str) -> list[str]:
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in filenames:
            if any(fn.endswith(ext) for ext in SOURCE_EXTENSIONS):
                files.append(os.path.join(root, fn))
    return files


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_gomi(repo_path: str, top_n: int = 10) -> None:
    repo_path = os.path.abspath(repo_path)

    print("\n" + "=" * 62)
    print("  GOMI — Technical Debt Risk Analyzer")
    print(f"  Repo : {repo_path}")
    print(f"  Window: last {ANALYSIS_WINDOW_DAYS} days (~6 months)")
    print(f"  Min commits for scoring: {MIN_COMMITS_FOR_SCORE}")
    print("=" * 62)

    if subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "--git-dir"],
        capture_output=True
    ).returncode != 0:
        print(f"ERROR: not a git repository: {repo_path}")
        sys.exit(1)

    timings: list[tuple[str, float]] = []

    # ── Layer 1: load fine-tuned DistilBERT ───────────────────────────────────
    print("\n[1/5] Loading DistilBERT sentiment model (OpenReview 2025)...")
    _t = time.perf_counter()
    sentiment_clf = load_sentiment_model()
    timings.append(("DistilBERT model load", time.perf_counter() - _t))

    # ── Layer 3 training: DeepJIT + ApacheJIT ─────────────────────────────────
    print("\n[2/5] Training risk model (DeepJIT + ApacheJIT)...")
    _t = time.perf_counter()
    risk_model, X_train = train_risk_model(sentiment_clf)
    timings.append(("Risk model train/load", time.perf_counter() - _t))

    # ── Git history (6-month window) ──────────────────────────────────────────
    print(f"\n[3/5] Extracting git history (last {ANALYSIS_WINDOW_DAYS} days)...")
    _t = time.perf_counter()
    git_stats = get_repo_git_stats(repo_path)
    timings.append(("Git history extract", time.perf_counter() - _t))
    print(f"  Git history covers {len(git_stats)} tracked files in window")

    # ── Layer 2: Lizard complexity on all source files ─────────────────────────
    print("\n[4/5] Scanning source files with Lizard...")
    _t = time.perf_counter()
    source_files = get_source_files(repo_path)
    print(f"  Found {len(source_files)} source files")
    if not source_files:
        print("  No source files found.")
        sys.exit(1)

    raw_complexity: dict[str, dict] = {}
    raw_todos: dict[str, dict] = {}
    for fp in source_files:
        rel = os.path.relpath(fp, repo_path)
        raw_complexity[rel] = compute_complexity(fp)
        raw_todos[rel] = count_todo_comments(fp)
    timings.append(("Lizard + TODO scan", time.perf_counter() - _t))

    # Repo-relative normalization: AvgCCN percentile rank within this repo
    all_ccn = [m["AvgCCN"] for m in raw_complexity.values()]

    # Change entropy: percentile-ranked within this repo (ent already in git_stats)
    all_ent_vals = [s.get("ent", 0.0) for s in git_stats.values()]

    # ── Score every file ──────────────────────────────────────────────────────
    print("\n[5/5] Computing risk scores...")
    _t = time.perf_counter()

    results = []
    for fp in source_files:
        rel    = os.path.relpath(fp, repo_path)
        stats  = git_stats.get(rel, {})
        msgs   = stats.get("messages", [])
        nf     = stats.get("nf", 0)

        # Low-confidence flag: fewer than MIN_COMMITS_FOR_SCORE commits in window
        low_confidence = nf < MIN_COMMITS_FOR_SCORE

        # Layer 1: DistilBERT sentiment score + low info ratio
        sentiment_score, low_info_ratio, breakdown = compute_sentiment_score(msgs, sentiment_clf)

        # Layer 2: Lizard complexity score (repo-relative percentile rank of AvgCCN)
        complexity_score = percentile_rank(raw_complexity[rel]["AvgCCN"], all_ccn)

        # Change entropy: percentile rank of git churn entropy within this repo.
        # Falls back to 0.0 when no git history exists (no churn → no entropy).
        change_entropy = (
            percentile_rank(stats.get("ent", 0.0), all_ent_vals)
            if all_ent_vals else 0.0
        )

        # Layer 3: Logistic Regression risk score — complexity is display-only
        satd_density = (
            float(np.mean([compute_msg_satd_density(m) for m in msgs]))
            if msgs else 0.0
        )
        X_file     = np.array([[sentiment_score, low_info_ratio, change_entropy, satd_density]])
        risk_score = float(risk_model.predict_proba(X_file)[0][1])

        # Layer 4: SHAP breakdown
        try:
            shap_out = compute_shap(risk_model, X_train, X_file)
        except Exception:
            shap_out = {
                "base_rate": 0.0,
                "sentiment_contrib": 0.0,
                "low_info_contrib": 0.0,
                "entropy_contrib": 0.0,
                "satd_contrib": 0.0,
            }

        results.append({
            "file":              rel,
            "sentiment_score":   sentiment_score,
            "complexity_score":  complexity_score,
            "low_info_ratio":    low_info_ratio,
            "change_entropy":    change_entropy,
            "satd_density":      satd_density,
            "risk_score":        risk_score,
            "shap":              shap_out,
            "commits":           breakdown,
            "complexity_raw":    raw_complexity[rel],
            "git_stats":         stats,
            "low_confidence":    low_confidence,
            "todo_counts":       raw_todos[rel],
        })

    timings.append(("Risk scoring + SHAP", time.perf_counter() - _t))

    results.sort(key=lambda x: x["risk_score"], reverse=True)

    # ─── OUTPUT ───────────────────────────────────────────────────────────────

    W = 62

    def line(text: str) -> None:
        print(f"│ {text:<{W - 2}} │")

    print("\n" + "=" * W)
    print(f"  RESULTS — {len(results)} files scored (showing top {min(top_n, len(results))})")
    print("=" * W)

    for r in results[:top_n]:
        score  = r["risk_score"]
        filled = int(score * 10)
        bar    = "█" * filled + "░" * (10 - filled)
        level  = (
            "HIGH RISK ⚠" if score >= 0.7 else
            "MODERATE  ~" if score >= 0.4 else
            "LOW       ✓"
        )
        flag = "  ⚑ low confidence" if r["low_confidence"] else ""
        print(f"\n  {r['file']:<40} [{bar}] {score:.2f}  {level}{flag}")

    if len(results) > top_n:
        print(f"\n  ... and {len(results) - top_n} more files (not shown)")

    print("\n" + "=" * W)
    print("  DRILLDOWN — top 5 riskiest files")
    print("=" * W)

    for r in results[:5]:
        print(f"\n┌{'─' * W}┐")
        line(f"FILE: {r['file']}")
        conf = "  ⚑ LOW CONFIDENCE (< 10 commits)" if r["low_confidence"] else ""
        line(f"Risk score: {r['risk_score']:.4f}{conf}")

        print(f"├{'─' * W}┤")
        shap_d = r["shap"]
        approx = (shap_d["base_rate"] + shap_d["sentiment_contrib"] +
                  shap_d["low_info_contrib"] + shap_d["entropy_contrib"] +
                  shap_d["satd_contrib"])
        line("SHAP breakdown (LR inputs only):")
        line(f"  base_rate          {shap_d['base_rate']:+.4f}")
        line(f"  sentiment_contrib  {shap_d['sentiment_contrib']:+.4f}"
             f"   (sentiment_score = {r['sentiment_score']:.3f})")
        line(f"  low_info_contrib   {shap_d['low_info_contrib']:+.4f}"
             f"   (low_info_ratio = {r['low_info_ratio']:.3f})")
        line(f"  entropy_contrib    {shap_d['entropy_contrib']:+.4f}"
             f"   (change_entropy = {r['change_entropy']:.3f})")
        line(f"  satd_contrib       {shap_d['satd_contrib']:+.4f}"
             f"   (satd_density = {r['satd_density']:.3f})")
        line(f"  {'─' * 28}")
        line(f"  ≈ risk score       {approx:+.4f}")

        print(f"├{'─' * W}┤")
        cr = r["complexity_raw"]
        line(
            f"Complexity: AvgCCN={cr['AvgCCN']:.1f}  "
            f"AvgNLOC={cr['AvgNLOC']:.1f}  "
            f"functions={cr['function_cnt']}  "
            f"PARAM={cr['PARAM']:.1f}"
        )

        print(f"├{'─' * W}┤")
        tc = r["todo_counts"]
        total_todos = sum(tc.values())
        if total_todos:
            breakdown_str = "  ".join(
                f"{tag}:{cnt}" for tag, cnt in tc.items() if cnt > 0
            )
            line(f"TODO comments: {total_todos}  ({breakdown_str})")
        else:
            line("TODO comments: 0")

        print(f"├{'─' * W}┤")
        gs = r["git_stats"]
        if gs:
            line("Git stats (ent → change_entropy feature; rest display only):")
            line(
                f"  la={gs['la']}  ld={gs['ld']}  commits={gs['nf']}  "
                f"devs={gs['ndev']}  age={gs['age_days']}d  ent={gs['ent']}"
            )
        else:
            line("Git stats: (no history in 6-month window for this file)")

        print(f"├{'─' * W}┤")
        line("Commit sentiment (DistilBERT):")
        if r["commits"]:
            for msg, emotion in r["commits"][:4]:
                short = (msg[:43] + "...") if len(msg) > 46 else msg
                line(f"  [{emotion:<12}] {short}")
        else:
            line("  (no commit messages in 6-month window)")

        print(f"└{'─' * W}┘")

    # ─── TIMING SUMMARY ───────────────────────────────────────────────────────

    total = sum(t for _, t in timings)
    print(f"\n{'─' * W}")
    print("  TIMING SUMMARY")
    print(f"{'─' * W}")
    for label, elapsed in timings:
        bar_w  = 20
        filled = round((elapsed / total) * bar_w) if total > 0 else 0
        bar    = "█" * filled + "░" * (bar_w - filled)
        print(f"  {label:<26} [{bar}]  {elapsed:>7.2f}s  ({elapsed / total * 100:4.1f}%)")
    print(f"{'─' * W}")
    print(f"  {'Total':<26}  {' ' * (bar_w + 2)}  {total:>7.2f}s")
    print(f"{'─' * W}\n")


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    top  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_gomi(repo, top_n=top)
