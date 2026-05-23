"""
label_5k.py — Apply trained DistilBERT sentiment model to 5k unlabeled commits.

Reads:  datasets/cleaned20k.csv          (full cleaned corpus)
        datasets/openreview_labeled_2k.csv (already labeled — excluded)
Model:  scripts/datasets/distilbert_sentiment

Writes: datasets/openreview_labeled_5k_auto.csv
        Columns: commit, author, date, repo, message, reconciled_emotion, confidence
"""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_DIR       = ROOT / "datasets"
SCRIPTS_DS_DIR    = ROOT / "scripts" / "datasets"
CLEANED_CSV       = DATASET_DIR / "cleaned20k.csv"
LABELED_CSV       = DATASET_DIR / "openreview_labeled_2k.csv"
MODEL_DIR         = SCRIPTS_DS_DIR / "distilbert_sentiment"
OUTPUT_CSV        = DATASET_DIR / "openreview_labeled_5k_auto.csv"

SAMPLE_SIZE  = 5000
BATCH_SIZE   = 64


def main():
    if not MODEL_DIR.is_dir():
        print(f"ERROR: model not found at {MODEL_DIR}")
        sys.exit(1)

    # Load commits already in 2k labeled set
    already_labeled: set[str] = set()
    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            already_labeled.add(row["commit"])
    print(f"Excluding {len(already_labeled)} already-labeled commits")

    # Collect unlabeled rows from cleaned20k
    candidates = []
    with open(CLEANED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["commit"] not in already_labeled and row.get("message", "").strip():
                candidates.append(row)
    print(f"Unlabeled candidates: {len(candidates)}")

    # Take first SAMPLE_SIZE (preserves original ordering)
    sample = candidates[:SAMPLE_SIZE]
    print(f"Sampling: {len(sample)}")

    # Load model
    print(f"\nLoading DistilBERT from {MODEL_DIR} ...")
    from transformers import pipeline as hf_pipeline
    classifier = hf_pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        top_k=None,
        truncation=True,
        max_length=128,
        device=-1,   # CPU; change to 0 for GPU
    )
    print("Model loaded.\n")

    # Run batch inference
    messages = [row["message"][:512] for row in sample]
    results  = []
    total    = len(messages)

    for start in range(0, total, BATCH_SIZE):
        batch = messages[start : start + BATCH_SIZE]
        preds = classifier(batch)
        results.extend(preds)
        done = min(start + BATCH_SIZE, total)
        print(f"  [{done}/{total}]", end="\r", flush=True)

    print(f"\nInference done. Writing {OUTPUT_CSV} ...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["commit", "author", "date", "repo",
                        "message", "reconciled_emotion", "confidence"],
        )
        writer.writeheader()

        label_counts: dict[str, int] = {}
        for row, pred_list in zip(sample, results):
            best   = max(pred_list, key=lambda x: x["score"])
            label  = best["label"].lower().replace("label_", "")
            conf   = round(best["score"], 4)
            label_counts[label] = label_counts.get(label, 0) + 1
            writer.writerow({
                "commit":             row["commit"],
                "author":             row["author"],
                "date":               row["date"],
                "repo":               row["repo"],
                "message":            row["message"],
                "reconciled_emotion": label,
                "confidence":         conf,
            })

    print(f"\nDone. Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<14} {count:>5}  ({100*count/len(sample):.1f}%)")
    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
