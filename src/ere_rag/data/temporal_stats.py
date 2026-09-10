"""
Inspect temporal relation statistics in MAVEN-ERE.(avergae number of temporal relations per event, distribution of relation types, etc.).

Usage :  python src/ere_rag/data/temporal_stats.py --data_path data/raw/MAVEN_ERE/train.jsonl
"""
import json
from collections import defaultdict

REL2ID = {
    "BEFORE": 0,
    "OVERLAP": 1,
    "CONTAINS": 2,
    "SIMULTANEOUS": 3,
    "ENDS-ON": 4,
    "BEGINS-ON": 5,
    "NONE": 6,
}

def load_docs(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def build_directed_graph(doc):
    out_adj = defaultdict(set)
    in_adj = defaultdict(set)
    n_positives = 0
        
    events = set()
    eid2mentions = {}
    if "events" in doc:
        for e in doc["events"]:
            events.add(e["mention"][k["id"]] for k in e["mention"])
            eid2mentions[e["id"]] = [
                m["id"] for m in e["mention"]
            ]
        relations = doc["temporal_relations"]
    else:
        events = doc["event_mentions"]
        relations = {}
    for t in doc["TIMEX"]:
        events.add(t["id"])
        eid2mentions[t["id"]] = [t["id"]]
    

    pair2rel = {}
    for rel in relations:
        for pair in relations[rel]:
            for e1 in eid2mentions[pair[0]]:
                for e2 in eid2mentions[pair[1]]:
                    pair2rel[(e1, e2)] = REL2ID[rel]
                    out_adj[e1].add(e2)
                    in_adj[e2].add(e1)
                    n_positives += 1
                    if rel in ["SIMULTANEOUS", "BEGINS-ON"]:
                        pair2rel[(e2, e1)] = REL2ID[rel]
                        out_adj[e2].add(e1)
                        in_adj[e1].add(e2)
                        n_positives += 1
    
    percentage_positive =  0.0
    for e in events:
        positives = 0
        if e in out_adj:
            positives = len(out_adj[e])
        percentage_positive += positives / (len(events) - 1) if len(events) > 1 else 0.0
    
    return n_positives ,len(events),n_positives/len(events), percentage_positive

def stats(path):
    total_events = 0
    total_docs = 0
    total_sum_out_adj = 0
   
    total_norm_out_adj = 0
    total_not_out = 0
    total_percentage_positive = 0.0
    for doc in load_docs(path):
        total_docs += 1
        sum_out_adj,  num_events, norm_out_adj, percentage_positives = build_directed_graph(doc)
        total_not_out += num_events**2 - num_events - sum_out_adj
        total_events += num_events
        total_sum_out_adj += sum_out_adj
        
        total_norm_out_adj += norm_out_adj
        total_percentage_positive += percentage_positives

    return {
       
        "avg_out": total_sum_out_adj/total_events,
        "avg_norm_degree": total_norm_out_adj/total_docs,
        "avg_not_out": total_not_out/total_events,
        "avg_events": total_events/total_docs,
        "avg_percentage_positive": total_percentage_positive/total_events,
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

    print("Temporal Relation Statistics:")
    for key, value in result.items():
        print(f"  {key}: {value}")
