"""
load_data.py

Reusable data loading utilities for MAVEN-ERE.
Imported by preprocessing, training, and evaluation scripts.

Usage as a script (for quick inspection):
    python src/ere_rag/data/load_data.py --data_dir data/raw
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# ── Relation type constants ────────────────────────────────────────────────

TEMPORAL_RELATION_TYPES = [
    "BEFORE", "OVERLAP", "CONTAINS", 
    "SIMULTANEOUS", "ENDS-ON", "BEGINS-ON"
]

CAUSAL_RELATION_TYPES = [
    "CAUSE", "PRECONDITION"
]

# ── Core loading functions ─────────────────────────────────────────────────

def load_jsonl(filepath):
    """
    Load a .jsonl file where each line is one JSON document.
    Returns a list of document dicts.
    """
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def load_split(data_dir, split='train'):
    """
    Load a dataset split by name.
    
    Args:
        data_dir: path to the directory containing .jsonl files
        split: one of 'train', 'valid', 'test'
    
    Returns:
        list of document dicts
    """
    filepath = Path(data_dir) / f"{split}.jsonl"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find {filepath}. "
            f"Make sure you downloaded MAVEN-ERE into {data_dir}."
        )
    return load_jsonl(filepath)


# ── Event and relation extraction helpers ─────────────────────────────────

def get_event_mentions(document):
    """
    Extract all event mentions from a document with their metadata.
    
    Each event in MAVEN-ERE is a coreference chain.
    This flattens the chain structure into individual mentions,
    keeping track of which chain each mention belongs to.
    
    Returns:
        list of dicts with keys:
            mention_id, trigger_word, sent_id, offset,
            event_id (chain id), event_type
    """
    mentions = []
    for event in document.get('events', []):
        event_id = event['id']
        event_type = event.get('type', 'UNKNOWN')
        for mention in event.get('mention', []):
            mentions.append({
                'mention_id': mention['id'],
                'trigger_word': mention['trigger_word'],
                'sent_id': mention['sent_id'],
                'offset': mention['offset'],
                'event_id': event_id,
                'event_type': event_type,
            })
    return mentions


def get_event_id_to_mentions(document):
    """
    Build a mapping from event chain ID to list of mention dicts.
    Useful for expanding event-level relations to mention-level.
    """
    mapping = defaultdict(list)
    for event in document.get('events', []):
        event_id = event['id']
        event_type = event.get('type', 'UNKNOWN')
        for mention in event.get('mention', []):
            mapping[event_id].append({
                'mention_id': mention['id'],
                'trigger_word': mention['trigger_word'],
                'sent_id': mention['sent_id'],
                'offset': mention['offset'],
                'event_id': event_id,
                'event_type': event_type,
            })
    return dict(mapping)


def get_temporal_relation_pairs(document):
    """
    Extract all temporal relation pairs from a document.
    Relations are stored at the event chain level in MAVEN-ERE.
    
    Returns:
        list of dicts with keys: head_id, tail_id, relation_type
        where head_id and tail_id are EVENT_ or TIME_ IDs
    """
    pairs = []
    temporal = document.get('temporal_relations', {})
    for rel_type in TEMPORAL_RELATION_TYPES:
        for pair in temporal.get(rel_type, []):
            pairs.append({
                'head_id': pair[0],
                'tail_id': pair[1],
                'relation_type': rel_type,
            })
    return pairs


def get_causal_relation_pairs(document):
    """
    Extract all causal relation pairs from a document.
    
    Returns:
        list of dicts with keys: head_id, tail_id, relation_type
        where relation_type is CAUSE or PRECONDITION
    """
    pairs = []
    causal = document.get('causal_relations', {})
    for rel_type in CAUSAL_RELATION_TYPES:
        for pair in causal.get(rel_type, []):
            pairs.append({
                'head_id': pair[0],
                'tail_id': pair[1],
                'relation_type': rel_type,
            })
    return pairs


def get_sentence(document, sent_id):
    """Get the text of a sentence by its index."""
    sentences = document.get('sentences', [])
    if sent_id < len(sentences):
        return sentences[sent_id]
    return ""


# ── Statistics ─────────────────────────────────────────────────────────────

def compute_statistics(documents):
    """
    Compute basic statistics for a list of documents.
    Returns a dict of statistics.
    """
    n_docs = len(documents)
    event_counts = []
    temporal_counts = []
    causal_counts = []
    possible_pair_counts = []

    temporal_type_dist = defaultdict(int)
    causal_type_dist = defaultdict(int)

    for doc in documents:
        mentions = get_event_mentions(doc)
        n_mentions = len(mentions)
        event_counts.append(n_mentions)
        possible_pair_counts.append(n_mentions * (n_mentions - 1) / 2)

        temporal_pairs = get_temporal_relation_pairs(doc)
        temporal_counts.append(len(temporal_pairs))
        for pair in temporal_pairs:
            temporal_type_dist[pair['relation_type']] += 1

        causal_pairs = get_causal_relation_pairs(doc)
        causal_counts.append(len(causal_pairs))
        for pair in causal_pairs:
            causal_type_dist[pair['relation_type']] += 1

    stats = {
        'n_documents': n_docs,
        'total_event_mentions': sum(event_counts),
        'avg_mentions_per_doc': sum(event_counts) / n_docs,
        'max_mentions_per_doc': max(event_counts),
        'total_temporal_relations': sum(temporal_counts),
        'avg_temporal_per_doc': sum(temporal_counts) / n_docs,
        'temporal_type_distribution': dict(temporal_type_dist),
        'total_causal_relations': sum(causal_counts),
        'avg_causal_per_doc': sum(causal_counts) / n_docs,
        'causal_type_distribution': dict(causal_type_dist),
        'avg_possible_pairs_per_doc': sum(possible_pair_counts) / n_docs,
    }
    return stats


def print_statistics(stats, split_name):
    """Pretty print statistics for a split."""
    print(f"\n{'='*50}")
    print(f"Statistics — {split_name} split")
    print(f"{'='*50}")
    print(f"Documents:                {stats['n_documents']}")
    print(f"Total event mentions:     {stats['total_event_mentions']}")
    print(f"Avg mentions/doc:         {stats['avg_mentions_per_doc']:.1f}")
    print(f"Max mentions in one doc:  {stats['max_mentions_per_doc']}")
    print(f"Avg possible pairs/doc:   {stats['avg_possible_pairs_per_doc']:.1f}")
    print(f"\nTemporal relations:       {stats['total_temporal_relations']}")
    print(f"Avg temporal/doc:         {stats['avg_temporal_per_doc']:.1f}")
    print(f"Types: {stats['temporal_type_distribution']}")
    print(f"\nCausal relations:         {stats['total_causal_relations']}")
    print(f"Avg causal/doc:           {stats['avg_causal_per_doc']:.1f}")
    print(f"Types: {stats['causal_type_distribution']}")


# ── Entry point for quick inspection ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Load and inspect MAVEN-ERE dataset'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/raw',
        help='Directory containing train.jsonl and valid.jsonl'
    )
    args = parser.parse_args()

    for split in ['train', 'valid']:
        try:
            print(f"\nLoading {split} split...")
            docs = load_split(args.data_dir, split)
            stats = compute_statistics(docs)
            print_statistics(stats, split)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")

    print("\nDone. These utilities are ready to import in other scripts.")


if __name__ == '__main__':
    main()