"""prepare_data.py

Builds grouped training examples for S-BERT fine-tuning.

Output format — one JSON line per (anchor, positive) pair:
{
    "anchor":      str,          # anchor event sentence with <m> markers
    "positive":    str,          # positive event sentence with <m> markers
    "negatives":   [str, ...],   # all negatives assigned to this pair
    "anchor_id":   str,
    "positive_id": str,
    "negative_ids": [str, ...],
    "doc_id":      str,
    "n_neg":       int           # len(negatives), stored for quick stats
}

Negative distribution strategy — anchor-centric, nothing wasted:

  For each anchor event e1:
    positives = all events annotated as related to e1 (temporal or causal)
    negatives = all events in the same document with NO annotation to e1

    The total pool of negatives is divided across positives:
      base      = n_neg // n_pos   negatives per positive (floor)
      remainder = n_neg %  n_pos   extra negatives for the first R positives

    Each (e1, e2) pair therefore gets a list of [base] or [base+1] negatives.
    ALL negatives are used exactly once across that anchor's positives.

  If n_neg < n_pos (rare), sample WITH replacement so every positive
  still gets exactly one negative.

  

Usage:
    python src/ere_rag/retriever/prepare_data.py \
        --data_dir data/raw/MAVEN_ERE \
        --output_dir data/interim \
        --sample_rate 0.15 \
        --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from ere_rag.baselines.temporal.data import Document

# ── Constants ──────────────────────────────────────────────────────────────

TEMPORAL_POSITIVE_IDS = {0, 1, 2, 3, 4, 5}
CAUSAL_TYPES          = {"CAUSE", "PRECONDITION"}
MARKER_OPEN           = "<m>"
MARKER_CLOSE          = "</m>"


# ── Text helpers ───────────────────────────────────────────────────────────

def wrap_event_in_sentence(document_data: dict, event: dict) -> str:
    """
    Reconstruct sentence from tokens[] and wrap the trigger word.
    Uses tokens[] (guaranteed field) not sentences[] (may be absent).
    """
    sent_id  = event.get("sent_id", 0)
    tokens   = document_data.get("tokens", [])
    sentence = " ".join(tokens[sent_id]) if sent_id < len(tokens) else ""
    trigger  = event.get("trigger_word") or event.get("mention", "")

    if trigger and trigger in sentence:
        return sentence.replace(
            trigger, f"{MARKER_OPEN} {trigger} {MARKER_CLOSE}", 1
        )
    return f"{sentence} {MARKER_OPEN} {trigger} {MARKER_CLOSE}".strip()


# ── Per-anchor map ─────────────────────────────────────────────────────────

def build_anchor_map(document_data: dict) -> dict[str, dict]:
    """
    For every event that has at least one annotated positive, build:
        anchor_id -> {
            "anchor":    event dict,
            "positives": [event dict, ...],
            "negatives": [event dict, ...],
        }

    Temporal relations come from Document.labels (includes TIMEX).
    Causal relations come from causal_relations[], expanded chain→mention.
    Both are pooled — retriever is binary.
    """
    doc    = Document(document_data, ignore_nonetype=False)
    events = doc.events
    n      = len(events)

    if n < 2:
        return {}

    # Collect positive (e1_id, e2_id) pairs
    positive_pairs: set[tuple[str, str]] = set()

    label_idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if doc.labels[label_idx] in TEMPORAL_POSITIVE_IDS:
                    positive_pairs.add((events[i]["id"], events[j]["id"]))
                label_idx += 1

    for rel_type, pairs in document_data.get("causal_relations", {}).items():
        if rel_type not in CAUSAL_TYPES:
            continue
        for pair in pairs:
            for m1 in doc.eid2mentions.get(pair[0], [pair[0]]):
                for m2 in doc.eid2mentions.get(pair[1], [pair[1]]):
                    if m1 != m2:
                        positive_pairs.add((m1, m2))

    # Build per-anchor split
    anchor_map: dict[str, dict] = {}
    for anchor in events:
        aid       = anchor["id"]
        positives = []
        negatives = []
        for other in events:
            oid = other["id"]
            if oid == aid:
                continue
            if (aid, oid) in positive_pairs:
                positives.append(other)
            else:
                negatives.append(other)
        if positives:
            anchor_map[aid] = {
                "anchor":    anchor,
                "positives": positives,
                "negatives": negatives,
            }

    return anchor_map


# ── Group construction ─────────────────────────────────────────────────────

def build_groups_for_document(
    document_data: dict,
    anchor_map:    dict[str, dict],
    rng:           random.Random,
) -> list[dict]:
    """
    For each (anchor, positive) pair, produce one group dict containing
    the pre-allocated slice of negatives for that pair.

    Case A  n_neg >= n_pos:
        Shuffle negatives. Distribute evenly:
          base      = n_neg // n_pos
          remainder = n_neg %  n_pos
        Positive i gets (base + 1) negatives if i < remainder, else base.
        All negatives used exactly once across the anchor's positives.

    Case B  n_neg < n_pos:
        Each positive gets exactly one negative sampled with replacement.
        Every positive still produces a group.
    """
    doc_id = document_data["id"]
    groups = []

    for aid, info in anchor_map.items():
        anchor    = info["anchor"]
        positives = info["positives"]
        negatives = info["negatives"]

        n_pos = len(positives)
        n_neg = len(negatives)

        if n_neg == 0:
            continue

        anchor_text = wrap_event_in_sentence(document_data, anchor)

        if n_neg >= n_pos:
            # Case A — distribute all negatives evenly, nothing wasted
            neg_shuffled = negatives[:]
            rng.shuffle(neg_shuffled)

            base      = n_neg // n_pos
            remainder = n_neg %  n_pos
            cursor    = 0

            for pos_idx, positive in enumerate(positives):
                n_for_this  = base + (1 if pos_idx < remainder else 0)
                assigned    = neg_shuffled[cursor : cursor + n_for_this]
                cursor     += n_for_this

                groups.append({
                    "anchor":       anchor_text,
                    "positive":     wrap_event_in_sentence(document_data, positive),
                    "negatives":    [wrap_event_in_sentence(document_data, ne)
                                     for ne in assigned],
                    "anchor_id":    aid,
                    "positive_id":  positive["id"],
                    "negative_ids": [ne["id"] for ne in assigned],
                    "doc_id":       doc_id,
                    "n_neg":        n_for_this,
                })

        else:
            # Case B — sample with replacement, one negative per positive
            for positive in positives:
                neg_event = rng.choice(negatives)
                groups.append({
                    "anchor":       anchor_text,
                    "positive":     wrap_event_in_sentence(document_data, positive),
                    "negatives":    [wrap_event_in_sentence(document_data, neg_event)],
                    "anchor_id":    aid,
                    "positive_id":  positive["id"],
                    "negative_ids": [neg_event["id"]],
                    "doc_id":       doc_id,
                    "n_neg":        1,
                })

    return groups


# ── I/O helpers ────────────────────────────────────────────────────────────

def load_and_sample_documents(
    jsonl_path:  Path,
    sample_rate: float,
    seed:        int,
) -> list[dict]:
    documents = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    total = len(documents)
    if sample_rate < 1.0:
        rng = random.Random(seed)
        n   = max(1, int(total * sample_rate))
        documents = rng.sample(documents, n)
        print(f"  Sampled {n:,} / {total:,} documents "
              f"(rate={sample_rate}, seed={seed})")
    else:
        print(f"  Loaded {total:,} documents")
    return documents


def save_groups(groups: list[dict], filepath: Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for g in groups:
            f.write(json.dumps(g) + "\n")
    print(f"  Saved {len(groups):,} groups → {filepath}")


def print_stats(groups: list[dict], label: str) -> None:
    n         = len(groups)
    n_docs    = len({g["doc_id"] for g in groups})
    n_anchors = len({(g["doc_id"], g["anchor_id"]) for g in groups})
    total_neg = sum(g["n_neg"] for g in groups)
    avg_neg   = total_neg / max(1, n)
    print(
        f"  [{label}]  groups={n:,}  docs={n_docs:,}  "
        f"anchors={n_anchors:,}  "
        f"total_negatives_stored={total_neg:,}  "
        f"avg_neg_per_group={avg_neg:.1f}"
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build grouped training examples for S-BERT retriever"
    )
    parser.add_argument("--data_dir",   type=str, default="data/raw/MAVEN_ERE")
    parser.add_argument("--output_dir", type=str, default="data/interim")
    parser.add_argument(
        "--sample_rate", type=float, default=0.15,
        help="Fraction of train documents to use (valid always uses all)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng        = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats      = {}

    for split in ["train", "valid"]:
        print(f"\nProcessing {split} split...")
        jsonl_path = Path(args.data_dir) / f"{split}.jsonl"

        if not jsonl_path.exists():
            print(f"  Not found: {jsonl_path}, skipping.")
            continue

        rate      = args.sample_rate if split == "train" else 1.0
        documents = load_and_sample_documents(jsonl_path, rate, args.seed)

        all_groups: list[dict] = []
        n_skipped = 0

        for doc_data in documents:
            try:
                anchor_map = build_anchor_map(doc_data)
            except Exception as e:
                print(f"  Warning: skipped {doc_data.get('id','?')} — {e}")
                n_skipped += 1
                continue

            if not anchor_map:
                n_skipped += 1
                continue

            all_groups.extend(
                build_groups_for_document(doc_data, anchor_map, rng)
            )

        if n_skipped:
            print(f"  Skipped {n_skipped} documents (no annotated positives)")

        print_stats(all_groups, split)
        save_groups(all_groups, output_dir / f"{split}_groups.jsonl")

        stats[split] = {
            "n_groups":          len(all_groups),
            "n_docs":            len({g["doc_id"] for g in all_groups}),
            "n_anchors":         len({(g["doc_id"], g["anchor_id"]) for g in all_groups}),
            "total_negatives":   sum(g["n_neg"] for g in all_groups),
        }

    with open(output_dir / "group_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats → {output_dir / 'group_stats.json'}")
    print(
        "\nEach line = one (anchor, positive, [negatives]) group.\n"
        "train_retriever.py samples --max_groups of these, then draws\n"
        "one negative per group randomly each epoch."
    )


if __name__ == "__main__":
    main()
