# adapt_vocab.py — Fast Vocabulary Transfer for commit message domain adaptation
#
# Cheaper CPU alternative to MLM pre-training. Identifies high-frequency commit
# message tokens that DistilBERT's WordPiece fragments into 2+ subwords, adds
# them to the vocabulary, and initializes their embeddings as the mean of their
# fragment embeddings (FVT). Touches only the embedding layer — no full retrain.
#
# Time: minutes on CPU.
# Output: datasets/distilbert_adapted/
#
# Run order:
#   1. python scripts/adapt_vocab.py       ← this script
#   2. python scripts/train_sentiment.py
#   3. python gomi.py <repo_path>
#
# Usage:
#   pip install transformers torch
#   python scripts/adapt_vocab.py

import csv
import os
import pickle
import re
import sys
from collections import Counter

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
OUTPUT_DIR  = os.path.join(DATASET_DIR, "distilbert_adapted")
BASE_MODEL  = "distilbert-base-uncased"

DEEPJIT_PKLS = [
    os.path.join(DATASET_DIR, f)
    for f in [
        "qt_test_raw.pkl", "openstack_test_raw.pkl", "go_test_raw.pkl",
        "jdt_test_raw.pkl", "gerrit_test_raw.pkl", "platform_test_raw.pkl",
    ]
]
APACHEJIT_CSV = os.path.join(DATASET_DIR, "apachejit_commits.csv")

MIN_FREQ       = 10     # minimum corpus frequency to add a token
MAX_NEW_TOKENS = 500    # cap additions — limits embedding matrix bloat
# Match words starting with a letter, min 3 chars (filters noise/numbers)
TOKEN_PATTERN  = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{2,}")


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
        print(f"  [deepjit/{os.path.basename(pkl_path).replace('_test_raw.pkl', '')}] {len(msgs)}")

    if os.path.isfile(APACHEJIT_CSV):
        apache_msgs = []
        with open(APACHEJIT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                msg = row.get("message", "").strip()
                if msg:
                    apache_msgs.append(msg)
        messages.extend(apache_msgs)
        print(f"  [apachejit] {len(apache_msgs)}")
    else:
        print(f"  [skip] apachejit: {APACHEJIT_CSV}")

    print(f"  Total corpus: {len(messages)} messages")
    return messages


def find_new_tokens(messages: list[str], tokenizer) -> list[str]:
    """
    Returns tokens that are high-frequency in the commit corpus but fragmented
    by DistilBERT's WordPiece into 2+ subwords, ordered by frequency desc.
    """
    word_freq: Counter = Counter()
    for msg in messages:
        word_freq.update(TOKEN_PATTERN.findall(msg.lower()))

    new_tokens = []
    for word, freq in word_freq.most_common():
        if freq < MIN_FREQ:
            break
        if len(tokenizer.tokenize(word)) > 1:
            new_tokens.append(word)
        if len(new_tokens) >= MAX_NEW_TOKENS:
            break

    return new_tokens


def adapt() -> None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install transformers torch")
        sys.exit(1)

    print("=" * 60)
    print("  GOMI — Vocabulary Adaptation (Fast Vocabulary Transfer)")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Min freq   : {MIN_FREQ}  |  Max new tokens: {MAX_NEW_TOKENS}")
    print("=" * 60)

    print("\n[1/4] Loading commit messages...")
    messages = load_commit_messages()
    if not messages:
        print("ERROR: No commit messages found. Check dataset paths.")
        sys.exit(1)

    print(f"\n[2/4] Loading DistilBERT tokenizer...")
    orig_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"\n[3/4] Finding fragmented domain tokens...")
    new_tokens = find_new_tokens(messages, orig_tokenizer)
    print(f"  Found {len(new_tokens)} tokens to add")
    if new_tokens:
        print(f"  Sample: {new_tokens[:10]}")

    if not new_tokens:
        print("\n  No new tokens found — corpus too small or already well-covered.")
        print("  Saving base model as-is.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        orig_tokenizer.save_pretrained(OUTPUT_DIR)
        AutoModelForMaskedLM.from_pretrained(BASE_MODEL).save_pretrained(OUTPUT_DIR)
        return

    print(f"\n[4/4] Applying FVT initialization...")
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    # new_tokenizer receives the additions; orig_tokenizer stays untouched for FVT lookup
    new_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    num_added = new_tokenizer.add_tokens(new_tokens)
    print(f"  Added {num_added} tokens (vocab size: {len(orig_tokenizer)} → {len(new_tokenizer)})")

    # Resize embedding matrix — new rows default to zero
    model.resize_token_embeddings(len(new_tokenizer))

    # FVT: initialize each new embedding as mean of its original fragment embeddings
    embedding_matrix = model.distilbert.embeddings.word_embeddings.weight
    initialized = 0
    with torch.no_grad():
        for token in new_tokens:
            new_id = new_tokenizer.convert_tokens_to_ids(token)
            if new_id == new_tokenizer.unk_token_id:
                continue
            frag_ids = orig_tokenizer.convert_tokens_to_ids(
                orig_tokenizer.tokenize(token)
            )
            if not frag_ids:
                continue
            embedding_matrix[new_id] = embedding_matrix[frag_ids].mean(dim=0)
            initialized += 1

    print(f"  FVT initialized {initialized}/{len(new_tokens)} embeddings")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    new_tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  Vocabulary adaptation complete.")
    print(f"  Model saved → {OUTPUT_DIR}")
    print("  Run train_sentiment.py next (auto-detects this model).")
    print("=" * 60)


if __name__ == "__main__":
    adapt()
