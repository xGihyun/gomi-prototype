# train_sentiment.py — One-time offline training script
#
# Fine-tunes distilbert-base-uncased on the OpenReview 2025 dataset
# (openreview_labeled_2k.csv) for 4-class commit message sentiment.
#
# Labels: frustration | caution | neutral | satisfaction
# Risk-positive labels (used by gomi.py): frustration, caution
#
# Output: saved HuggingFace model + tokenizer at datasets/distilbert_sentiment/
#         This directory is what gomi.py loads at runtime — run this script once.
#
# Usage:
#   pip install transformers datasets scikit-learn torch
#   python train_sentiment.py
#
# Expected CSV columns in openreview_labeled_2k.csv:
#   message               — the raw commit message text
#   reconciled_emotion    — one of: frustration, caution, neutral, satisfaction

import csv
import os
import sys

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ─── PATHS ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, "datasets")
CSV_PATH     = os.path.join(DATASET_DIR, "openreview_labeled_2k.csv")
OUTPUT_DIR   = os.path.join(DATASET_DIR, "distilbert_sentiment")

_MLM_PRETRAINED = os.path.join(DATASET_DIR, "distilbert_commit_mlm")
BASE_MODEL = _MLM_PRETRAINED if os.path.isdir(_MLM_PRETRAINED) else "distilbert-base-uncased"

# ─── LABEL SCHEME ─────────────────────────────────────────────────────────────

# Canonical label order — must stay consistent between training and gomi.py
LABELS      = ["frustration", "caution", "neutral", "satisfaction"]
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
ID2LABEL    = {i: l for i, l in enumerate(LABELS)}

VALID_EMOTIONS = set(LABELS)

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────

MAX_LENGTH   = 128      # commit messages rarely exceed 128 tokens
BATCH_SIZE   = 16
NUM_EPOCHS   = 4        # 4 epochs on 2k samples; adjust if val loss plateaus
LEARNING_RATE = 2e-5   # standard for DistilBERT fine-tuning
WEIGHT_DECAY  = 0.01
TEST_SIZE     = 0.15    # 15% held out for evaluation reporting (not used by gomi.py)
RANDOM_SEED   = 42

# ─── LOAD DATASET ─────────────────────────────────────────────────────────────

def load_openreview(csv_path: str) -> tuple[list[str], list[int]]:
    """
    Reads openreview_labeled_2k.csv and returns (messages, label_ids).
    Rows with missing/invalid labels are skipped with a warning.
    """
    if not os.path.isfile(csv_path):
        print(f"\nERROR: Dataset not found: {csv_path}")
        print(f"Download the OpenReview 2025 labeled commit dataset and place it at:")
        print(f"  {csv_path}")
        print(f"  Source: https://openreview.net/forum?id=FPLNSx1jmL\n")
        sys.exit(1)

    messages, label_ids = [], []
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg     = row.get("message", "").strip()
            emotion = row.get("reconciled_emotion", "").strip().lower()
            if not msg or emotion not in VALID_EMOTIONS:
                skipped += 1
                continue
            messages.append(msg)
            label_ids.append(LABEL2ID[emotion])

    print(f"  Loaded {len(messages)} rows  ({skipped} skipped — missing/invalid label)")

    # Print label distribution so class imbalance is visible
    from collections import Counter
    dist = Counter(LABELS[i] for i in label_ids)
    for label, count in sorted(dist.items()):
        pct = 100 * count / len(label_ids)
        print(f"    {label:<14} {count:>4}  ({pct:.1f}%)")

    return messages, label_ids


# ─── DATA AUGMENTATION ────────────────────────────────────────────────────────

def augment_with_nlpaug(
    messages: list[str], labels: list[int]
) -> tuple[list[str], list[int]]:
    """
    Returns new (augmented) samples via contextual word embedding substitution.
    Uses distilbert-base-uncased as the substitution model (15% token swap).
    Falls back gracefully if nlpaug is not installed.
    """
    try:
        import nlpaug.augmenter.word as naw
    except ImportError:
        print("  [nlpaug] not installed — skipping (pip install nlpaug)")
        return [], []

    aug = naw.ContextualWordEmbsAug(
        model_path="distilbert-base-uncased",
        action="substitute",
        aug_p=0.15,
        device="cpu",
    )

    aug_msgs, aug_labels = [], []
    print(f"  nlpaug: augmenting {len(messages)} samples...")
    for msg, label in zip(messages, labels):
        try:
            result = aug.augment(msg)
            aug_msgs.append(result[0] if isinstance(result, list) else result)
            aug_labels.append(label)
        except Exception:
            pass

    print(f"  nlpaug: +{len(aug_msgs)} samples generated")
    return aug_msgs, aug_labels


def back_translate(
    messages: list[str], labels: list[int]
) -> tuple[list[str], list[int]]:
    """
    Returns back-translated samples via MarianMT (en→de→en).
    Downloads Helsinki-NLP models on first call (~300 MB each).
    Falls back gracefully if models are unavailable.
    """
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        print("  [back-translation] transformers/torch not installed — skipping")
        return [], []

    print("  back-translation: loading MarianMT en↔de models...")
    en2de_name = "Helsinki-NLP/opus-mt-en-de"
    de2en_name = "Helsinki-NLP/opus-mt-de-en"

    tok_en2de = MarianTokenizer.from_pretrained(en2de_name)
    mdl_en2de = MarianMTModel.from_pretrained(en2de_name)
    tok_de2en = MarianTokenizer.from_pretrained(de2en_name)
    mdl_de2en = MarianMTModel.from_pretrained(de2en_name)

    def translate(texts, tokenizer, model):
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        )
        with torch.no_grad():
            out = model.generate(**inputs)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in out]

    BATCH = 32
    bt_msgs = []
    print(f"  back-translation: translating {len(messages)} samples (batch={BATCH})...")
    for i in range(0, len(messages), BATCH):
        batch = messages[i : i + BATCH]
        try:
            de = translate(batch, tok_en2de, mdl_en2de)
            en = translate(de, tok_de2en, mdl_de2en)
            bt_msgs.extend(en)
        except Exception as e:
            print(f"    [warn] batch {i // BATCH} failed: {e}")
            bt_msgs.extend(batch)

    print(f"  back-translation: +{len(bt_msgs)} samples generated")
    return bt_msgs, list(labels)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train():
    # ── Import heavy deps here so the file can be read without them installed ──
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"\nERROR: Missing dependency — {e}")
        print("Install with: pip install transformers datasets torch scikit-learn\n")
        sys.exit(1)

    print("=" * 60)
    print("  GOMI — DistilBERT Sentiment Fine-tuning")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Dataset    : {CSV_PATH}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Epochs     : {NUM_EPOCHS}  |  LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading OpenReview 2025 dataset...")
    messages, label_ids = load_openreview(CSV_PATH)

    train_msgs, val_msgs, train_labels, val_labels = train_test_split(
        messages, label_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=label_ids,      # preserve class balance in both splits
    )
    print(f"  Train: {len(train_msgs)}  |  Val: {len(val_msgs)}")

    # ── Data augmentation (training split only) ───────────────────────────────
    print("\n[1b/4] Augmenting training data...")
    nlp_msgs, nlp_labels = augment_with_nlpaug(train_msgs, train_labels)
    bt_msgs,  bt_labels  = back_translate(train_msgs, train_labels)
    train_msgs   = train_msgs   + nlp_msgs   + bt_msgs
    train_labels = train_labels + nlp_labels + bt_labels
    print(f"  Final training set: {len(train_msgs)} samples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading tokenizer ({BASE_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = Dataset.from_dict({"text": train_msgs, "label": train_labels})
    val_ds   = Dataset.from_dict({"text": val_msgs,   "label": val_labels})

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize,   batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds   = val_ds.remove_columns(["text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Loading {BASE_MODEL} and attaching classification head...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── LoRA (optional — reduces overfitting risk on 2k samples) ──────────────
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        _use_lora = True
        print("  [LoRA] enabled — training adapter weights only")
    except ImportError:
        _use_lora = False
        print("  [LoRA] peft not installed — full fine-tuning")
        print("  Install with: pip install peft")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Fine-tuning for {NUM_EPOCHS} epochs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(
            labels, preds,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy":  report["accuracy"],
            "f1_macro":  report["macro avg"]["f1-score"],
            "precision": report["macro avg"]["precision"],
            "recall":    report["macro avg"]["recall"],
        }

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="no" if _use_lora else "epoch",
        load_best_model_at_end=False if _use_lora else True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",       # disable wandb/mlflow
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── Final evaluation (before merge so trainer.model still valid) ──────────
    print("\nFinal evaluation on validation set:")
    preds_out = trainer.predict(val_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    print(classification_report(
        val_labels, preds,
        target_names=LABELS,
        zero_division=0,
    ))

    # ── Merge LoRA weights into base model before saving ──────────────────────
    if _use_lora:
        print("  Merging LoRA adapter weights into base model...")
        model = model.merge_and_unload()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving fine-tuned model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model saved → {OUTPUT_DIR}")
    print("  You can now run:  python gomi.py <repo_path>")
    print("=" * 60)


if __name__ == "__main__":
    train()
