"""
preprocess.py

Preprocesses MAVEN-ERE for the ere_rag pipeline.

What this script does:
1. Loads train.jsonl and valid.jsonl from data/raw/MAVEN_ERE/
2. Creates a 15% random-seed subset of each split
3. Extracts event pairs with temporal and causal relation labels
4. Extracts negative pairs (same document, no annotated relation)
5. Saves everything to data/processed/

Usage:
    python src/ere_rag/data/preprocess.py
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

from load_data import (
    load_split,
    get_event_mentions,
    get_event_id_to_mentions,
    get_temporal_relation_pairs,
    get_causal_relation_pairs,
    get_sentence,
    TEMPORAL_RELATION_TYPES,
    CAUSAL_RELATION_TYPES,
)

# ── Constants ──────────────────────────────────────────────────────────────

RANDOM_SEED = 42
SUBSET_FRACTION = 0.15
NO_RELATION_LABEL = "NOREL"

# How many hard negatives to sample per document
# Too many makes training data huge, too few hurts the retriever
NEGATIVES_PER_POSITIVE = 3


# ── Subsetting ─────────────────────────────────────────────────────────────

def create_subset(documents, fraction, seed):
    """
    Sample a fixed random subset of documents.
    
    Args:
        documents: full list of document dicts
        fraction: proportion to keep e.g. 0.15
        seed: random seed for reproducibility
    
    Returns:
        subset list of document dicts
    """
    rng = random.Random(seed)
    subset_size = max(1, int(len(documents) * fraction))
    subset = rng.sample(documents, subset_size)
    print(f"  Subset: {subset_size} / {len(documents)} documents "
          f"({fraction*100:.0f}%, seed={seed})")
    return subset


# ── Event pair extraction ──────────────────────────────────────────────────

def get_annotated_relation_pairs(document):
    """
    Get all annotated positive relation pairs at the MENTION level.
    
    MAVEN-ERE stores relations at the event chain level.
    We expand them to mention level because:
    - The retriever works on mention embeddings
    - Predictions are submitted at mention level
    
    For each event chain pair with a relation, we create one pair
    using the first mention of each chain (following common practice).
    
    Returns:
        list of dicts with keys:
            head_mention_id, head_trigger, head_sent_id,
            tail_mention_id, tail_trigger, tail_sent_id,
            relation_type, relation_category (temporal/causal),
            head_sentence, tail_sentence,
            doc_id
    """
    doc_id = document['id']
    event_id_to_mentions = get_event_id_to_mentions(document)
    pairs = []

    def add_pairs(raw_pairs, category):
        for raw_pair in raw_pairs:
            head_event_id = raw_pair['head_id']
            tail_event_id = raw_pair['tail_id']
            rel_type = raw_pair['relation_type']

            # Skip if either ID is a TIMEX (starts with TIME_)
            # We focus on event-event relations only
            if (head_event_id.startswith('TIME_') or
                    tail_event_id.startswith('TIME_')):
                continue

            head_mentions = event_id_to_mentions.get(head_event_id, [])
            tail_mentions = event_id_to_mentions.get(tail_event_id, [])

            if not head_mentions or not tail_mentions:
                continue

            # Use first mention of each chain
            head = head_mentions[0]
            tail = tail_mentions[0]

            pairs.append({
                'doc_id': doc_id,
                'head_mention_id': head['mention_id'],
                'head_trigger': head['trigger_word'],
                'head_sent_id': head['sent_id'],
                'head_sentence': get_sentence(document, head['sent_id']),
                'tail_mention_id': tail['mention_id'],
                'tail_trigger': tail['trigger_word'],
                'tail_sent_id': tail['sent_id'],
                'tail_sentence': get_sentence(document, tail['sent_id']),
                'relation_type': rel_type,
                'relation_category': category,
                'label': rel_type,
            })

    add_pairs(get_temporal_relation_pairs(document), 'temporal')
    add_pairs(get_causal_relation_pairs(document), 'causal')

    return pairs


def get_negative_pairs(document, positive_pairs, n_per_positive, seed):
    """
    Sample hard negative pairs from the same document.
    
    Hard negatives: event mention pairs from the same document
    that have NO annotated relation. These are harder to distinguish
    than random negatives because they share document context.
    
    Args:
        document: document dict
        positive_pairs: list of already-extracted positive pairs
        n_per_positive: how many negatives to sample per positive pair
        seed: random seed
    
    Returns:
        list of dicts in same format as positive pairs but with NOREL label
    """
    doc_id = document['id']
    rng = random.Random(seed)

    # Collect all mention IDs involved in positive pairs
    positive_pair_set = set()
    for pair in positive_pairs:
        positive_pair_set.add(
            (pair['head_mention_id'], pair['tail_mention_id'])
        )
        # Also add reverse to avoid symmetric duplicates
        positive_pair_set.add(
            (pair['tail_mention_id'], pair['head_mention_id'])
        )

    # Get all event mentions in this document
    mentions = get_event_mentions(document)
    if len(mentions) < 2:
        return []

    # Generate all possible mention pairs not in positive set
    candidate_negatives = []
    for m1, m2 in combinations(mentions, 2):
        pair_key = (m1['mention_id'], m2['mention_id'])
        reverse_key = (m2['mention_id'], m1['mention_id'])
        if pair_key not in positive_pair_set and \
                reverse_key not in positive_pair_set:
            candidate_negatives.append((m1, m2))

    # Sample a limited number
    n_to_sample = min(
        len(candidate_negatives),
        n_per_positive * len(positive_pairs)
    )
    sampled = rng.sample(candidate_negatives, n_to_sample)

    negative_pairs = []
    for m1, m2 in sampled:
        negative_pairs.append({
            'doc_id': doc_id,
            'head_mention_id': m1['mention_id'],
            'head_trigger': m1['trigger_word'],
            'head_sent_id': m1['sent_id'],
            'head_sentence': get_sentence(document, m1['sent_id']),
            'tail_mention_id': m2['mention_id'],
            'tail_trigger': m2['trigger_word'],
            'tail_sent_id': m2['sent_id'],
            'tail_sentence': get_sentence(document, m2['sent_id']),
            'relation_type': NO_RELATION_LABEL,
            'relation_category': 'none',
            'label': NO_RELATION_LABEL,
        })

    return negative_pairs


# ── Saving ─────────────────────────────────────────────────────────────────

def save_jsonl(data, filepath):
    """Save a list of dicts as a .jsonl file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(data)} records to {filepath}")


def save_json(data, filepath):
    """Save a dict as a .json file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {filepath}")


# ── Statistics ─────────────────────────────────────────────────────────────

def compute_pair_statistics(pairs, split_name):
    """Print statistics about extracted pairs."""
    total = len(pairs)
    temporal = [p for p in pairs if p['relation_category'] == 'temporal']
    causal = [p for p in pairs if p['relation_category'] == 'causal']
    negative = [p for p in pairs if p['label'] == NO_RELATION_LABEL]

    temporal_types = defaultdict(int)
    for p in temporal:
        temporal_types[p['relation_type']] += 1

    causal_types = defaultdict(int)
    for p in causal:
        causal_types[p['relation_type']] += 1

    print(f"\n--- Pair statistics for {split_name} ---")
    print(f"Total pairs:       {total}")
    print(f"Temporal positive: {len(temporal)}")
    for t, c in temporal_types.items():
        print(f"  {t}: {c}")
    print(f"Causal positive:   {len(causal)}")
    for t, c in causal_types.items():
        print(f"  {t}: {c}")
    print(f"Negative (NOREL):  {len(negative)}")
    if len(temporal) + len(causal) > 0:
        ratio = len(negative) / (len(temporal) + len(causal))
        print(f"Negative/positive ratio: {ratio:.2f}")

    return {
        'total_pairs': total,
        'temporal_positive': len(temporal),
        'causal_positive': len(causal),
        'negative': len(negative),
        'temporal_type_distribution': dict(temporal_types),
        'causal_type_distribution': dict(causal_types),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def process_split(documents, split_name, output_dir, seed):
    """
    Process one split: extract pairs, sample negatives, save.
    """
    print(f"\nProcessing {split_name} split ({len(documents)} documents)...")

    all_pairs = []
    docs_skipped = 0

    for doc in documents:
        # Get positive pairs
        positive_pairs = get_annotated_relation_pairs(doc)

        if not positive_pairs:
            docs_skipped += 1
            # Still include the document but with no pairs
            continue

        # Get negative pairs
        negative_pairs = get_negative_pairs(
            doc, positive_pairs, NEGATIVES_PER_POSITIVE, seed
        )

        all_pairs.extend(positive_pairs)
        all_pairs.extend(negative_pairs)

    print(f"  Documents skipped (no relations): {docs_skipped}")

    # Save pairs
    output_path = Path(output_dir) / f"{split_name}_pairs.jsonl"
    save_jsonl(all_pairs, output_path)

    # Compute and return statistics
    stats = compute_pair_statistics(all_pairs, split_name)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess MAVEN-ERE for ere_rag pipeline'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/raw/MAVEN_ERE',
        help='Path to MAVEN_ERE folder containing train.jsonl and valid.jsonl'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/processed',
        help='Where to save processed files'
    )
    parser.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help='Random seed for subsetting and negative sampling'
    )
    parser.add_argument(
        '--fraction', type=float, default=SUBSET_FRACTION,
        help='Fraction of documents to keep in subset'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_stats = {}

    for split in ['train', 'valid']:
        print(f"\n{'='*50}")
        print(f"Loading {split} split from {args.data_dir}...")

        try:
            documents = load_split(args.data_dir, split)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Create subset
        print(f"Creating {args.fraction*100:.0f}% subset...")
        subset = create_subset(documents, args.fraction, args.seed)

        # Save subset documents
        subset_path = output_dir / f"subset_{split}.jsonl"
        save_jsonl(subset, subset_path)

        # Extract and save pairs
        stats = process_split(subset, split, output_dir, args.seed)
        all_stats[split] = stats

    # Save overall statistics
    stats_path = output_dir / "stats.json"
    save_json(all_stats, stats_path)

    print(f"\n{'='*50}")
    print("Preprocessing complete.")
    print(f"Output files in: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()