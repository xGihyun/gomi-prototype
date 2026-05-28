# pretrain_mlm.py — Domain adaptive MLM pre-training on commit messages
#
# Pre-trains distilbert-base-uncased on raw commit messages from DeepJIT
# and ApacheJIT before fine-tuning on labeled OpenReview data.
# Output saved to datasets/distilbert_commit_mlm/, which train_sentiment.py
# picks up automatically if present.
#
# Run order:
#   1. python scripts/pretrain_mlm.py      ← this script
#   2. python scripts/train_sentiment.py
#   3. python gomi.py <repo_path>
#
# Usage:
#   pip install transformers datasets torch
#   python scripts/pretrain_mlm.py

import csv
import os
import pickle
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
OUTPUT_DIR  = os.path.join(DATASET_DIR, "distilbert_commit_mlm")
BASE_MODEL  = "distilbert-base-uncased"

DEEPJIT_PKLS = [
    os.path.join(DATASET_DIR, f)
    for f in [
        "qt_test_raw.pkl", "openstack_test_raw.pkl", "go_test_raw.pkl",
        "jdt_test_raw.pkl", "gerrit_test_raw.pkl", "platform_test_raw.pkl",
    ]
]
APACHEJIT_CSV = os.path.join(DATASET_DIR, "apachejit_commits.csv")

MLM_PROBABILITY = 0.15
MAX_LENGTH      = 128
BATCH_SIZE      = 32
NUM_STEPS       = 2000   # ~1 epoch over 60k commits at batch 32
LEARNING_RATE   = 5e-5
RANDOM_SEED     = 42


def load_commit_messages() -> list[str]:
    messages = []

    for pkl_path in DEEPJIT_PKLS:
        if not os.path.isfile(pkl_path):
            project = os.path.basename(pkl_path).replace("_test_raw.pkl", "")
            print(f"  [skip] {project}: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        msgs = [m for m in raw[2] if m and m.strip()]
        messages.extend(msgs)
        print(f"  [deepjit/{os.path.basename(pkl_path).replace('_test_raw.pkl','')}] {len(msgs)} messages")

    if os.path.isfile(APACHEJIT_CSV):
        apache_msgs = []
        with open(APACHEJIT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                msg = row.get("message", "").strip()
                if msg:
                    apache_msgs.append(msg)
        messages.extend(apache_msgs)
        print(f"  [apachejit] {len(apache_msgs)} messages")
    else:
        print(f"  [skip] apachejit: {APACHEJIT_CSV}")

    print(f"  Total corpus: {len(messages)} messages")
    return messages


def pretrain() -> None:
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForMaskedLM,
            DataCollatorForLanguageModeling,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install transformers datasets torch")
        sys.exit(1)

    print("=" * 60)
    print("  GOMI — DistilBERT MLM Domain Pre-training")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Steps      : {NUM_STEPS}  |  LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}")
    print("=" * 60)

    print("\n[1/3] Loading commit messages...")
    messages = load_commit_messages()
    if not messages:
        print("ERROR: No commit messages found. Check dataset paths.")
        sys.exit(1)

    print(f"\n[2/3] Tokenizing {len(messages)} messages...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    dataset = Dataset.from_dict({"text": messages})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    print(f"\n[3/3] Pre-training MLM for {NUM_STEPS} steps...")
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        max_steps=NUM_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        save_steps=NUM_STEPS,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    print(f"\nSaving pre-trained model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  MLM pre-training complete.")
    print(f"  Model saved → {OUTPUT_DIR}")
    print("  Run train_sentiment.py next (auto-detects this model).")
    print("=" * 60)


if __name__ == "__main__":
    pretrain()
