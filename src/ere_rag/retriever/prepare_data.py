"""prepare_data.py

Builds grouped training examples for S-BERT fine-tuning, for one of
three categories selected via --category:
    temporal  — positive pairs from MAVEN-ERE temporal relations.
                Anchor set = event mentions + TIMEX (temporal Document).
    causal    — positive pairs from MAVEN-ERE causal_relations
                (CAUSE/PRECONDITION). Anchor set = event mentions ONLY —
                no TIMEX.
    both      — union of temporal and causal positive pairs. Anchor set
                = the temporal Document's full event list (mentions +
                TIMEX), since temporal relations need TIMEX. A TIMEX
                anchor will simply never get a causal positive — causal
                relations never reference TIMEX ids, so this is expected.

Uses TWO Document classes ( existing ones, unmodified):
    ere_rag.baselines.temporal.data.Document  — for temporal positives
                                                  and the "both"/"temporal"
                                                  anchor set (includes TIMEX)
    ere_rag.baselines.causal.data.Document    — for causal positives
                                                  (handles chain→mention
                                                  expansion via its own
                                                  eid2mentions/get_labels,
                                                  no symmetric expansion,
                                                  since CAUSE/PRECONDITION
                                                  are directional)

Output format — one JSON line per (anchor, positive) pair:
{
    "anchor":          str,
    "positive":        str,
    "negatives":       [str, ...],   # ALL negatives assigned to this pair
                                      # (even distribution, unchanged)
    "anchor_id":       str,
    "positive_id":     str,
    "negative_ids":    [str, ...],
    "doc_id":          str,
    "n_neg":           int,
    "n_events_in_doc": int,           # kept for future use (e.g. percentage-
                                       # based retrieval at inference time) —
                                       
    "category":        str            # "temporal" | "causal" | "both"
}

Negative distribution strategy — anchor-centric, nothing wasted (unchanged):
for each anchor, ALL in-document negatives are evenly split across its
positives so no negative is wasted. How many of those assigned negatives
actually get sampled into training triplets is decided in
train_retriever.py via --k_negatives 

Usage:
    python src/ere_rag/retriever/prepare_data.py \
        --category causal \
        --data_dir data/raw/MAVEN_ERE \
        --output_dir data/interim/causal \
        --sample_rate 0.15 \
        --seed 42
"""

import json
import random
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
from ere_rag.baselines.temporal.data import Document as TemporalDocument
from ere_rag.baselines.causal.data import Document as CausalDocument

# ── Constants ──────────────────────────────────────────────────────────────

TEMPORAL_POSITIVE_IDS = {0, 1, 2, 3, 4, 5}    # BEFORE..BEGINS-ON, excludes NONE=6
CAUSAL_POSITIVE_IDS   = {0, 1}                 # CAUSE, PRECONDITION, excludes NONE=2
MARKER_OPEN            = "<m>"
MARKER_CLOSE           = "</m>"
VALID_CATEGORIES       = ("temporal", "causal", "both")


# ── Text helpers ───────────────────────────────────────────────────────────

def wrap_event_in_sentence(document_data: dict, event: dict) -> str:
    """
    Reconstruct sentence from tokens[] and wrap the trigger word.
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


# ── Positive-pair extraction, per category ──────────────────────────────────

def get_positive_pairs_temporal(doc: TemporalDocument) -> set[tuple[str, str]]:
    """Same logic doc.get_labels() already uses, re-derived as an id-pair set."""
    events = doc.events
    n      = len(events)
    positive_pairs: set[tuple[str, str]] = set()
    label_idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if doc.labels[label_idx] in TEMPORAL_POSITIVE_IDS:
                    positive_pairs.add((events[i]["id"], events[j]["id"]))
                label_idx += 1
    return positive_pairs


def get_positive_pairs_causal(doc: CausalDocument) -> set[tuple[str, str]]:
    """
    Mirrors get_positive_pairs_temporal but against the causal Document's
    own .events/.labels — these already reflect chain→mention expansion
    (via eid2mentions in causal data.py's get_labels()).
    """
    events = doc.events
    n      = len(events)
    positive_pairs: set[tuple[str, str]] = set()
    label_idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if doc.labels[label_idx] in CAUSAL_POSITIVE_IDS:
                    positive_pairs.add((events[i]["id"], events[j]["id"]))
                label_idx += 1
    return positive_pairs


# ── Per-anchor map ─────────────────────────────────────────────────────────

def build_anchor_map(document_data: dict, category: str) -> tuple[dict[str, dict], int]:
    """
    For every event that has at least one annotated positive (under the
    given category), build:
        anchor_id -> {
            "anchor":    event dict,
            "positives": [event dict, ...],
            "negatives": [event dict, ...],
        }

    Also returns n, the total number of events in the anchor set used
    
    """
    if category == "causal":
        c_doc  = CausalDocument(document_data, ignore_nonetype=False)
        events = c_doc.events
        n      = len(events)
        if n < 2:
            return {}, n
        positive_pairs = get_positive_pairs_causal(c_doc)

    else:
        # "temporal" and "both" both anchor on the temporal Document's
        # event set (includes TIMEX).
        t_doc  = TemporalDocument(document_data, ignore_nonetype=False)
        events = t_doc.events
        n      = len(events)
        if n < 2:
            return {}, n
        positive_pairs = get_positive_pairs_temporal(t_doc)

        if category == "both":
            c_doc = CausalDocument(document_data, ignore_nonetype=False)
            if len(c_doc.events) >= 2:
                positive_pairs = positive_pairs | get_positive_pairs_causal(c_doc)

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

    return anchor_map, n


# ── Group construction ───────────────────────────────────────────────────────

def build_groups_for_document(
    document_data:   dict,
    anchor_map:       dict[str, dict],
    rng:              random.Random,
    n_events_in_doc:  int,
    category:         str,
) -> list[dict]:
    """
    Case A  n_neg >= n_pos: distribute all negatives evenly across positives.
    Case B  n_neg < n_pos: sample with replacement, one negative per positive.
    (Unchanged — category-agnostic, just tags the result.)
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
                    "anchor":          anchor_text,
                    "positive":        wrap_event_in_sentence(document_data, positive),
                    "negatives":       [wrap_event_in_sentence(document_data, ne)
                                        for ne in assigned],
                    "anchor_id":       aid,
                    "positive_id":     positive["id"],
                    "negative_ids":    [ne["id"] for ne in assigned],
                    "doc_id":          doc_id,
                    "n_neg":           n_for_this,
                    "n_events_in_doc": n_events_in_doc,
                    "category":        category,
                })

        else:
            for positive in positives:
                neg_event = rng.choice(negatives)
                groups.append({
                    "anchor":          anchor_text,
                    "positive":        wrap_event_in_sentence(document_data, positive),
                    "negatives":       [wrap_event_in_sentence(document_data, neg_event)],
                    "anchor_id":       aid,
                    "positive_id":     positive["id"],
                    "negative_ids":    [neg_event["id"]],
                    "doc_id":          doc_id,
                    "n_neg":           1,
                    "n_events_in_doc": n_events_in_doc,
                    "category":        category,
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
    parser.add_argument(
        "--category", type=str, required=True, choices=VALID_CATEGORIES,
        help="Which relation type(s) to build positive pairs from."
    )
    parser.add_argument("--data_dir",   type=str, default="data/raw/MAVEN_ERE")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--sample_rate", type=float, default=0.15,
        help="Fraction of train documents to use (valid always uses all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Data-prep seed — keep this fixed (42) across all runs of a "
             "given category. Only train_retriever.py's --seed should vary."
    )
    args = parser.parse_args()

    rng        = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats      = {}

    print(f"Category: {args.category}")

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
                anchor_map, n_events_in_doc = build_anchor_map(doc_data, args.category)
            except Exception as e:
                print(f"  Warning: skipped {doc_data.get('id','?')} — {e}")
                n_skipped += 1
                continue

            if not anchor_map:
                n_skipped += 1
                continue

            all_groups.extend(
                build_groups_for_document(doc_data, anchor_map, rng, n_events_in_doc, args.category)
            )

        if n_skipped:
            print(f"  Skipped {n_skipped} documents (no annotated positives)")

        print_stats(all_groups, f"{split}/{args.category}")
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


if __name__ == "__main__":
    main()