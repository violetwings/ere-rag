"""
preprocess.py

Preprocesses MAVEN-ERE for the ere_rag pipeline, producing
valid_pairs.jsonl (and train_pairs.jsonl) in data/processed/.

These files are consumed by build_index.py for retrieval evaluation.

Design decisions — derived from the two baseline data.py files:

  Temporal pairs (temporal/data.py):
    - Event pool = event mentions + TIMEX mentions (time expressions).
      The baseline adds every TIMEX to the event list; omitting them
      would miss all temporal relations involving "yesterday", "2003", etc.
    - Relations expanded from event-chain level to all mention pairs
      (not just first mention per chain).
    - SIMULTANEOUS and BEGINS-ON are symmetric: (e1,e2) and (e2,e1)
      both get the relation label, matching baseline behaviour.
    - Pairs with abs(sent_id diff) > 1 are labelled NOREL (not -100),
      because we need concrete negative examples for retrieval evaluation.

  Causal pairs (causal/data.py):
    - Event pool = event mentions only (no TIMEX).
    - Relations: CAUSE and PRECONDITION only. NONE otherwise.
    - No symmetry expansion.

  Sentences:
    - Reconstructed by joining the token list for the event's sentence,
      exactly as the baseline tokenizer does. This is more reliable than
      using a pre-built "sentences" field which may be absent or
      inconsistently formatted.

Output schema per row (same for temporal and causal):
    {
      "doc_id":            str,
      "head_mention_id":   str,
      "head_trigger":      str,          # trigger_word for events, mention for TIMEX
      "head_sent_id":      int,
      "head_sentence":     str,          # tokens[sent_id] joined
      "tail_mention_id":   str,
      "tail_trigger":      str,
      "tail_sent_id":      int,
      "tail_sentence":     str,
      "relation_type":     str,          # e.g. "BEFORE", "CAUSE", "NOREL"
      "relation_category": str,          # "temporal" or "causal"
      "label":             str           # same as relation_type; "NOREL" for negatives
    }

Usage:
    python src/ere_rag/data/preprocess.py \
        --data_dir data/raw/MAVEN_ERE \
        --output_dir data/processed 
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# ── Relation constants ─────────────────────────────────────────────────────

TEMPORAL_RELS = {"BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS", "ENDS-ON", "BEGINS-ON"}
# Symmetric temporal relations — (e2,e1) gets the same label as (e1,e2)
TEMPORAL_SYMMETRIC = {"SIMULTANEOUS", "BEGINS-ON"}

CAUSAL_RELS = {"CAUSE", "PRECONDITION"}

NOREL = "NOREL"


# ── Sentence reconstruction ────────────────────────────────────────────────

def get_sentence(document_data: dict, sent_id: int) -> str:
    """
    Reconstruct a sentence by joining the token list for that sentence.

    This matches what the baseline tokenizers do: they iterate over
    document_data["tokens"] (a list of lists of strings, one per sentence)
    and join with spaces. Using a pre-built "sentences" field is unreliable
    because it may be absent or formatted differently across dataset versions.
    """
    tokens = document_data.get("tokens", [])
    if sent_id < len(tokens):
        return " ".join(tokens[sent_id])
    return ""


def get_trigger(event: dict) -> str:
    """
    Return the trigger word for an event mention or TIMEX.
    Event mentions use 'trigger_word'; TIMEX entries use 'mention'.
    """
    return event.get("trigger_word") or event.get("mention", "")


# ── Event pool construction ────────────────────────────────────────────────

def build_event_pool_temporal(document_data: dict) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Build the temporal event pool: event mentions + TIMEX.

    Matches temporal/data.py Document.__init__:
      for e in data["events"]: pool += e["mention"]
      for t in data["TIMEX"]:  pool.append(t)

    Returns:
      events:       sorted list of mention/TIMEX dicts
      eid2mentions: event-chain-id -> list of mention ids
    """
    events = []
    eid2mentions: dict[str, list[str]] = {}

    if "events" in document_data:
        for chain in document_data["events"]:
            for mention in chain["mention"]:
                events.append(mention)
            eid2mentions[chain["id"]] = [m["id"] for m in chain["mention"]]

    for timex in document_data.get("TIMEX", []):
        events.append(timex)
        eid2mentions[timex["id"]] = [timex["id"]]

    # Sort by sentence then token offset, matching baseline sort_events()
    events.sort(key=lambda e: (e["sent_id"], e["offset"][0]))
    return events, eid2mentions


def build_event_pool_causal(document_data: dict) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Build the causal event pool: event mentions only (no TIMEX).

    Matches causal/data.py Document.__init__:
      for e in data["events"]: pool += e["mention"]
      (no TIMEX addition)
    """
    events = []
    eid2mentions: dict[str, list[str]] = {}

    if "events" in document_data:
        for chain in document_data["events"]:
            for mention in chain["mention"]:
                events.append(mention)
            eid2mentions[chain["id"]] = [m["id"] for m in chain["mention"]]

    events.sort(key=lambda e: (e["sent_id"], e["offset"][0]))
    return events, eid2mentions


# ── Label map construction ─────────────────────────────────────────────────

def build_temporal_label_map(
    document_data: dict,
    eid2mentions: dict[str, list[str]],
) -> dict[tuple[str, str], str]:
    """
    Build (mention_id, mention_id) -> relation_type for temporal relations.

    Expands chain-level pairs to all mention-level pairs (not just first
    mention), matching baseline get_labels() behaviour. Applies symmetry
    for SIMULTANEOUS and BEGINS-ON.
    """
    pair2rel: dict[tuple[str, str], str] = {}
    relations = document_data.get("temporal_relations", {})

    for rel_type, pairs in relations.items():
        if rel_type not in TEMPORAL_RELS:
            continue
        for chain_pair in pairs:
            eid1, eid2 = chain_pair[0], chain_pair[1]
            for m1 in eid2mentions.get(eid1, [eid1]):
                for m2 in eid2mentions.get(eid2, [eid2]):
                    pair2rel[(m1, m2)] = rel_type
                    if rel_type in TEMPORAL_SYMMETRIC:
                        pair2rel[(m2, m1)] = rel_type

    return pair2rel


def build_causal_label_map(
    document_data: dict,
    eid2mentions: dict[str, list[str]],
) -> dict[tuple[str, str], str]:
    """
    Build (mention_id, mention_id) -> relation_type for causal relations.

    No symmetry expansion — causal relations are directional.
    """
    pair2rel: dict[tuple[str, str], str] = {}
    relations = document_data.get("causal_relations", {})

    for rel_type, pairs in relations.items():
        if rel_type not in CAUSAL_RELS:
            continue
        for chain_pair in pairs:
            eid1, eid2 = chain_pair[0], chain_pair[1]
            for m1 in eid2mentions.get(eid1, [eid1]):
                for m2 in eid2mentions.get(eid2, [eid2]):
                    pair2rel[(m1, m2)] = rel_type

    return pair2rel


# ── Pair extraction ────────────────────────────────────────────────────────

def extract_pairs_from_document(
    document_data: dict,
    category: str,          # "temporal" or "causal"
) -> list[dict]:
    """
    Extract all ordered event-mention pairs from one document,
    labelling each as a named relation type or NOREL.

    For temporal: includes TIMEX events, all mention expansions,
                  symmetric labels for SIMULTANEOUS/BEGINS-ON.
    For causal:   event mentions only, directional relations only.

    Returns a flat list of pair dicts in the schema described in the
    module docstring.
    """
    doc_id = document_data["id"]

    if category == "temporal":
        events, eid2mentions = build_event_pool_temporal(document_data)
        pair2rel = build_temporal_label_map(document_data, eid2mentions)
    else:
        events, eid2mentions = build_event_pool_causal(document_data)
        pair2rel = build_causal_label_map(document_data, eid2mentions)

    if len(events) < 2:
        return []

    pairs = []
    for e1 in events:
        for e2 in events:
            if e1["id"] == e2["id"]:
                continue

            rel = pair2rel.get((e1["id"], e2["id"]), NOREL)

            pairs.append({
                "doc_id":            doc_id,
                "head_mention_id":   e1["id"],
                "head_trigger":      get_trigger(e1),
                "head_sent_id":      e1["sent_id"],
                "head_sentence":     get_sentence(document_data, e1["sent_id"]),
                "tail_mention_id":   e2["id"],
                "tail_trigger":      get_trigger(e2),
                "tail_sent_id":      e2["sent_id"],
                "tail_sentence":     get_sentence(document_data, e2["sent_id"]),
                "relation_type":     rel,
                "relation_category": category,
                "label":             rel,
            })

    return pairs


# ── I/O helpers ────────────────────────────────────────────────────────────

def load_jsonl(filepath: Path) -> list[dict]:
    docs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def save_jsonl(rows: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"    Saved {len(rows):,} rows → {filepath}")


def print_stats(pairs: list[dict], label: str) -> None:
    total    = len(pairs)
    positive = sum(1 for p in pairs if p["label"] != NOREL)
    norel    = total - positive
    temporal = sum(1 for p in pairs if p["relation_category"] == "temporal" and p["label"] != NOREL)
    causal   = sum(1 for p in pairs if p["relation_category"] == "causal"   and p["label"] != NOREL)

    # Relation type breakdown
    by_type: dict[str, int] = defaultdict(int)
    for p in pairs:
        if p["label"] != NOREL:
            by_type[p["label"]] += 1

    print(f"  [{label}]  total={total:,}  positive={positive:,}  norel={norel:,}")
    print(f"    temporal positives: {temporal:,}  |  causal positives: {causal:,}")
    for rel, count in sorted(by_type.items()):
        print(f"    {rel}: {count:,}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MAVEN-ERE → data/processed/[train|valid]_pairs.jsonl"
    )
    parser.add_argument("--data_dir",   type=str, default="data/raw/MAVEN_ERE")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict = {}

    for split in ["train", "valid"]:
        print(f"\n{'='*55}")
        print(f"Split: {split}")
        jsonl_path = Path(args.data_dir) / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"  Not found: {jsonl_path}, skipping.")
            continue

        documents = load_jsonl(jsonl_path)
        print(f"  Loaded {len(documents):,} documents")

        temporal_pairs: list[dict] = []
        causal_pairs:   list[dict] = []
        skipped = 0

        for doc in documents:
            try:
                tp = extract_pairs_from_document(doc, "temporal")
                cp = extract_pairs_from_document(doc, "causal")
            except Exception as e:
                print(f"  Warning: skipped {doc.get('id','?')} — {e}")
                skipped += 1
                continue

            temporal_pairs.extend(tp)
            causal_pairs.extend(cp)

        if skipped:
            print(f"  Skipped {skipped} documents due to errors")

        # Merge: one combined file (used by build_index.py)
        all_pairs = temporal_pairs + causal_pairs

        print_stats(all_pairs, split)

        save_jsonl(all_pairs,      output_dir / f"{split}_pairs.jsonl")
        save_jsonl(temporal_pairs, output_dir / f"{split}_temporal_pairs.jsonl")
        save_jsonl(causal_pairs,   output_dir / f"{split}_causal_pairs.jsonl")

        n_pos_t = sum(1 for p in temporal_pairs if p["label"] != NOREL)
        n_pos_c = sum(1 for p in causal_pairs   if p["label"] != NOREL)
        all_stats[split] = {
            "n_documents":           len(documents) - skipped,
            "temporal_pairs":        len(temporal_pairs),
            "temporal_positives":    n_pos_t,
            "causal_pairs":          len(causal_pairs),
            "causal_positives":      n_pos_c,
            "total_pairs":           len(all_pairs),
            "total_positives":       n_pos_t + n_pos_c,
        }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats → {output_dir / 'stats.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()