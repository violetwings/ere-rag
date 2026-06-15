"""
pipeline.py

End-to-end ERE pipeline: Load cached retrieval → LLM classification.
Supports Ollama (local M1/M2), Claude API, and HuggingFace models.

Usage: python src/ere_rag/llm/pipeline.py \
    --run_dir experiments/retriever/<run_name> \
    --split valid \
    --n_docs 1 \
    --k 15 \
    --partition_size 15 \
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from ere_rag.llm.inference import load_model, run_inference
from ere_rag.llm.parse_output import parse_llm_output, predictions_to_pairs

# ── Data Loaders ───────────────────────────────────────────────────────────

def load_jsonl(filepath: Path) -> list[dict]:
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r") as f:
        return json.load(f)

def partition_candidates(candidates: list[dict], partition_size: int = 5) -> list[list[dict]]:
    return [candidates[i : i + partition_size] for i in range(0, len(candidates), partition_size)]

# ── Prompt Builder (Inline) ────────────────────────────────────────────────

def build_prompt_from_cache(query_event: dict, candidates: list[dict]) -> str:
    """Build LLM prompt from cached candidates using the Expert Persona layout."""
    format_lines = "\n".join([f"  e{i}: LABEL | reason" for i in range(len(candidates))])

    candidates_text = ""
    for i, cand in enumerate(candidates):
        candidates_text += f"e{i}: {cand['trigger']}\n    Sentence: {cand['wrapped_sentence']}\n"

    query_sentence = query_event.get('wrapped_sentence', query_event.get('sentence', ''))

    prompt = f"""You are an expert linguist performing Event Relation Extraction (ERE). 
Your task is to classify the temporal and causal relationship between a "Query" event and several "Candidate" events using the provided context.

CRITICAL RULES:
1. MARKER TAGS: The exact event triggers are enclosed in <m> and </m> tags. You MUST evaluate the relationship between these specific marked words, even if they share the exact same sentence. Ignore other unmarked events.
2. For each candidate, output TWO lines — one for temporal, one for causal:
  e{i}_temporal: LABEL | reason
  e{i}_causal: LABEL | reason
Use NOREL on a line only if that relation type genuinely does not exist.
BEFORE/OVERLAP/CONTAINS/SIMULTANEOUS/BEGINS-ON/ENDS-ON are temporal only.
CAUSE/PRECONDITION are causal only.

RELATION DEFINITIONS:
- CAUSE: The query event directly causes or results in the candidate event.
- PRECONDITION: The query event is required for the candidate to happen, but doesn't directly cause it.
- BEFORE: The query event finishes completely before the candidate event starts.
- OVERLAP: The events share some time, but one doesn't completely contain the other.
- CONTAINS: The query event's timeframe completely surrounds the candidate event.
- SIMULTANEOUS: The events occur at the exact same time.
- BEGINS-ON: The query event starts at the exact same time as the candidate event.
- ENDS-ON: The query event finishes at the exact same time as the candidate event.
- NOREL: There is no direct temporal or causal relationship, or they are unrelated background events.

--- TARGET EVENTS ---
Query event (head): <m>{query_event['trigger_word']}</m>
Query sentence: {query_sentence}

Candidate events (tails) to classify:
{candidates_text.rstrip()}

--- TASK ---
Format your response exactly as:
{format_lines}
"""
    return prompt

# ── Evaluation Math ────────────────────────────────────────────────────────

def _get_cat(label: str) -> str:
    temporal = {"BEFORE", "OVERLAP", "CONTAINS", "SIMULTANEOUS", "ENDS-ON", "BEGINS-ON"}
    return "temporal" if label in temporal else "causal"

def evaluate_predictions(predictions, pairs):
    # gt: (head_id, tail_id, category) -> label
    gt = {}
    for pair in pairs:
        if pair["label"] != "NOREL":
            key = (pair["head_mention_id"], pair["tail_mention_id"],
                   pair["relation_category"])
            gt[key] = pair["label"]

    # pred: same triple -> predicted_label
    pred = {}
    for p in predictions:
        key = (p["head_mention_id"], p["tail_mention_id"],
               p.get("relation_category", "temporal"))
        pred[key] = p.get("predicted_label", "NOREL")

    def counts(filter_cat=None):
        tp = fp = fn = 0
        relevant = {k for k in gt if filter_cat is None or k[2] == filter_cat}
        for key in relevant:
            pred_label = pred.get(key, "NOREL")
            if pred_label == gt[key]:
                tp += 1
            elif pred_label != "NOREL":
                fp += 1; fn += 1
            else:
                fn += 1
        for key, pred_label in pred.items():
            if pred_label != "NOREL" and key not in gt:
                if filter_cat is None or key[2] == filter_cat:
                    fp += 1
        p_ = tp/(tp+fp) if (tp+fp) else 0.0
        r_ = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = 2*p_*r_/(p_+r_) if (p_+r_) else 0.0
        return {"precision": round(p_,4), "recall": round(r_,4), "f1": round(f1,4)}

    return {
        "overall":  counts(),
        "temporal": counts("temporal"),
        "causal":   counts("causal"),
    }

# ── Pipeline Execution ─────────────────────────────────────────────────────

def run_document(
    doc_data: dict,
    doc_pairs: list[dict],
    doc_id: str,
    cache_dict: dict,
    k: int,
    partition_size: int,
    timing_totals: dict,
) -> tuple[list[dict], list[dict]]:
    if not cache_dict:
        return [], []

    gt_head_ids = {p["head_mention_id"] for p in doc_pairs if p["label"] != "NOREL"}
    all_predictions: list[dict] = []
    timing_rows: list[dict] = []

    for query_id in gt_head_ids:
        
        cached_entry = cache_dict.get(query_id)
        if not cached_entry:
            continue

        query_event = {
            "id":              query_id,
            "trigger_word":    cached_entry["query_trigger"],
            "wrapped_sentence": cached_entry["query_sentence"],
        }
        all_candidates = cached_entry["candidates"]
        top_k_candidates = all_candidates[:k]

        
        if not top_k_candidates:
            continue

        partitions = partition_candidates(top_k_candidates, partition_size)

        for part_idx, partition in enumerate(partitions):
            
            prompt = build_prompt_from_cache(query_event, partition)
            response, llm_elapsed = run_inference(prompt)

            parsed = parse_llm_output(response, len(partition), partition)
            pred_pairs = predictions_to_pairs(query_event, parsed, doc_id)

            all_predictions.extend(pred_pairs)
            timing_totals["llm_total"] = timing_totals.get("llm_total", 0) + llm_elapsed
            timing_totals["llm_n_calls"] = timing_totals.get("llm_n_calls", 0) + 1

            timing_rows.append({
                "doc_id": doc_id,
                "query_id": query_id,
                "partition": part_idx,
                "n_candidates": len(partition),
                "llm_s": round(llm_elapsed, 3),
            })

    return all_predictions, timing_rows

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM-based ERE pipeline using cached retrieval")
    parser.add_argument("--run_dir", type=Path, required=True, help="Experiment run directory")
    parser.add_argument("--data_dir", type=Path, default="data/raw/MAVEN_ERE", help="Path to raw documents")
    parser.add_argument("--processed_dir", type=Path, default="data/processed", help="Path to processed pairs")
    parser.add_argument("--llm_backend", type=str, default="ollama",
                        choices=["ollama", "api", "small", "local"],
                        help="LLM backend choice")
    parser.add_argument("--llm_model", type=str, default="llama3",
                        help="Model name ('llama3' for ollama, or HuggingFace IDs)")
    parser.add_argument("--split", type=str, default="valid", help="Data split (valid/test)")
    parser.add_argument("--n_docs", type=int, default=None, help="Limit to N documents")
    parser.add_argument("--k", type=int, default=5, help="Use top-k cached candidates")
    parser.add_argument("--partition_size", type=int, default=5, help="Batch size for LLM")
    args = parser.parse_args()
    cache_dir = args.run_dir / "cache"
    output_dir = args.run_dir / f"smoke_test_{args.llm_model}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LLM backend: {args.llm_backend} (model: {args.llm_model})")
    load_model(backend=args.llm_backend, model_name=args.llm_model, device="cpu")

    print(f"Loading documents from {args.data_dir}...")
    raw_docs = {d["id"]: d for d in load_jsonl(Path(args.data_dir) / f"{args.split}.jsonl")}

    print(f"Loading pairs from {args.processed_dir}...")
    all_pairs = load_jsonl(Path(args.processed_dir) / f"{args.split}_pairs.jsonl")
    pairs_by_doc: dict[str, list[dict]] = defaultdict(list)
    for pair in all_pairs:
        pairs_by_doc[pair["doc_id"]].append(pair)

    doc_ids = [doc_id for doc_id, doc_pairs in pairs_by_doc.items() if any(p["label"] != "NOREL" for p in doc_pairs)]
    if args.n_docs:
        doc_ids = doc_ids[: args.n_docs]

    print(f"Running on {len(doc_ids)} documents (backend={args.llm_backend}, k={args.k})\n")

    all_predictions, all_timing, timing_totals = [], [], {}
    pipeline_start = time.time()

    for i, doc_id in enumerate(tqdm(doc_ids, desc="Pipeline")):
        doc_start = time.time()
        cache_path = cache_dir / f"{doc_id}_cache.json"
        cache_dict = load_cache(cache_path)

        if not cache_dict:
            continue

        doc_data = raw_docs.get(doc_id)
        doc_pairs = pairs_by_doc.get(doc_id, [])

        preds, t_rows = run_document(doc_data, doc_pairs, doc_id, cache_dict, args.k, args.partition_size, timing_totals)
        all_predictions.extend(preds)
        all_timing.extend(t_rows)

    pipeline_elapsed = time.time() - pipeline_start

    # ── Evaluate ───────────────────────────────────────────────────────
    print("\nEvaluating...")
    evaluated_pairs = [p for p in all_pairs if p["doc_id"] in doc_ids]
    metrics = evaluate_predictions(all_predictions, evaluated_pairs)

    # ── Save Results ───────────────────────────────────────────────────
    with open(output_dir / "predictions.jsonl", "w") as f:
        for pred in all_predictions: f.write(json.dumps(pred) + "\n")

    with open(output_dir / "timing.jsonl", "w") as f:
        for row in all_timing: f.write(json.dumps(row) + "\n")

    results = {
        "cache_dir": str(cache_dir),
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
        "k": args.k,
        "n_docs": len(doc_ids),
        "metrics": metrics,
        "timing": {
            "total_s": round(pipeline_elapsed, 1),
            "per_doc_s": round(pipeline_elapsed / max(1, len(doc_ids)), 2),
            "llm_n_calls": timing_totals.get("llm_n_calls", 0),
            "avg_llm_call_s": round(timing_totals.get("llm_total", 0) / max(1, timing_totals.get("llm_n_calls", 1)), 2),
        }
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print Output ───────────────────────────────────────────────────
    print("\n=== PIPELINE RESULTS ===")
    print(f"Documents processed: {len(doc_ids)}")
    print(f"Total predictions:   {len(all_predictions)}")
    print(f"Total wall time:     {pipeline_elapsed:.1f}s")
    
    print("\n           P        R        F1")
    for cat in ["overall", "temporal", "causal"]:
        m = metrics[cat]
        print(f"  {cat:<10} {m['precision']:.4f}  {m['recall']:.4f}  {m['f1']:.4f}")

    print(f"\nResults saved to → {output_dir / 'results.json'}")

if __name__ == "__main__":
    main()