"""
Inspect causal relation statistics in MAVEN-ERE.(average number of causal relations per event, distribution of relation types, etc.).

Usage :  python src/ere_rag/data/causal_stats.py --data_path data/raw/MAVEN_ERE/train.jsonl
"""
import json
from collections import defaultdict

def load_docs(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def build_directed_graph(doc):
    out_adj = defaultdict(set)
    in_adj = defaultdict(set)
    

    if "causal_relations" not in doc:
        return out_adj, in_adj

    for rel_type, pairs in doc["causal_relations"].items():
        for e1, e2 in pairs:
            out_adj[e1].add(e2)
            in_adj[e2].add(e1)

    return out_adj, in_adj

def stats(path):
    all_in, all_out, all_norm_degrees, all_not_out = [], [], [], []
    total_events = 0
    total_docs = 0
    for doc in load_docs(path):
        out_adj, in_adj = build_directed_graph(doc)

        events = set()
        if "events" in doc:
            for e in doc["events"]:
                events.add(e["id"])
        else:
            for e in doc["event_mentions"]:
                events.add(e["id"])
        n = len(events)
        total_docs += 1
        total_events += n
        for e in events:
            all_out.append(len(out_adj[e]))
            all_not_out.append(n - 1 - len(out_adj[e]))
            all_in.append(len(in_adj[e]))
            if n > 1:
                all_norm_degrees.append(len(out_adj[e]) / (n - 1))

    return {
        "avg_in": sum(all_in)/len(all_in),
        "avg_out": sum(all_out)/len(all_out),
        "max_in": max(all_in),
        "max_out": max(all_out),
        "min_in": min(all_in),
        "min_out": min(all_out),
        "avg_norm_degree": sum(all_norm_degrees)/len(all_norm_degrees),
        "max_norm_degree": max(all_norm_degrees),
        "min_norm_degree": min(all_norm_degrees),
        "avg_not_out": sum(all_not_out)/len(all_not_out),
        "max_not_out": max(all_not_out),
        "min_not_out": min(all_not_out),
        "avg_events": total_events/total_docs,
    }
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/MAVEN_ERE/train.jsonl",
        help="Path to MAVEN-ERE jsonl file"
    )

    args = parser.parse_args()

    result = stats(args.data_path)

    print("Causal Relation Statistics:")
    for key, value in result.items():
        print(f"  {key}: {value}")
