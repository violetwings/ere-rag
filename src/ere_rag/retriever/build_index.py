"""
build_index.py

Encodes all event mentions in the VALIDATION split using the
fine-tuned S-BERT model and evaluates retriever recall@k for RQ1.

Performance design:
  - All mentions in a document are encoded in ONE batched model.encode()
    call. The resulting embedding matrix is stored and reused for both
    index construction and all query lookups.
  - FAISS search runs ONCE per query at max(k_values) and results are
    sliced for smaller k — eliminates redundant searches.
  - No per-query model.encode() calls inside any loop.

With 710 validation documents on Apple M1 this takes ~3-5 minutes
instead of the previous 45-60 minutes.

Usage:
    python src/ere_rag/retriever/build_index.py \
        --output_dir experiments/retriever/<run_name> \
        --processed_dir data/processed \
        --k_values 5 10 20 30 50
"""

import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
import pickle
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Device ─────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Marker tokens — must match prepare_data.py and train_retriever.py ──────

MARKER_OPEN  = "<m>"
MARKER_CLOSE = "</m>"


# ── Text helpers ────────────────────────────────────────────────────────────

def wrap_trigger(sentence: str, trigger: str) -> str:
    if trigger and trigger in sentence:
        return sentence.replace(trigger, f"{MARKER_OPEN} {trigger} {MARKER_CLOSE}", 1)
    return f"{sentence} {MARKER_OPEN} {trigger} {MARKER_CLOSE}".strip()


# ── Data loading ────────────────────────────────────────────────────────────

def load_jsonl(filepath: Path) -> list[dict]:
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Metrics ─────────────────────────────────────────────────────────────────

def prf(tp: int, retrieved: int, relevant: int) -> tuple[float, float, float]:
    p  = tp / retrieved if retrieved > 0 else 0.0
    r  = tp / relevant  if relevant  > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ── Per-document evaluation ──────────────────────────────────────────────────

def evaluate_document(
    mentions:    list[dict],          # all unique mentions in the document
    embeddings:  np.ndarray,          # shape (n_mentions, dim), already normalised
    gt:          dict[str, set],      # head_id -> set of tail_ids
    gt_cat:      dict[str, str],      # head_id -> "temporal" or "causal"
    mention_idx: dict[str, int],      # mention_id -> position in mentions list
    k_values:    list[int],
    max_k:       int,
) -> tuple[dict, dict, dict]:
    """
    Evaluate one document using pre-computed embeddings.

    Runs FAISS search once per query at max_k, then slices for all k values.
    No model.encode() calls here — embeddings are passed in from the batch.

    Returns accumulators for overall, temporal, causal — each a dict
    keyed by k with {"tp", "retrieved", "total"}.
    """
    n = len(mentions)
    if n < 2:
        return {}, {}, {}

    # Build flat inner-product index from pre-computed embeddings
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    acc          = {k: {"tp": 0, "retrieved": 0, "total": 0} for k in k_values}
    acc_temporal = {k: {"tp": 0, "retrieved": 0, "total": 0} for k in k_values}
    acc_causal   = {k: {"tp": 0, "retrieved": 0, "total": 0} for k in k_values}

    for hid, true_tails in gt.items():
        q_idx = mention_idx.get(hid)
        if q_idx is None:
            continue

        query_emb = embeddings[q_idx : q_idx + 1]   # shape (1, dim), already normalised
        cat       = gt_cat.get(hid, "temporal")

        # Single FAISS search at max_k — reuse results for all k values
        search_k       = min(max_k + 1, n)
        _, raw_indices = index.search(query_emb, search_k)

        # Build ordered list of retrieved IDs excluding the query itself
        retrieved_all: list[str] = []
        for idx in raw_indices[0]:
            cand_id = mentions[idx]["head_mention_id"]
            if cand_id != hid:
                retrieved_all.append(cand_id)
            if len(retrieved_all) >= max_k:
                break

        # Slice for each k — no extra search needed
        for k in k_values:
            retrieved_ids = set(retrieved_all[:k])
            tp = len(true_tails & retrieved_ids)

            acc[k]["tp"]        += tp
            acc[k]["retrieved"] += len(retrieved_ids)
            acc[k]["total"]     += len(true_tails)

            target = acc_temporal if cat == "temporal" else acc_causal
            target[k]["tp"]        += tp
            target[k]["retrieved"] += len(retrieved_ids)
            target[k]["total"]     += len(true_tails)

    return acc, acc_temporal, acc_causal


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(
    pairs:     list[dict],
    model:     SentenceTransformer,
    k_values:  list[int],
    output_dir: Path,
    batch_size: int = 256,
) -> dict:
    """
    Evaluate retriever recall@k across all documents.

    Key optimisation: for each document, encode ALL mentions in one
    batched model.encode() call, store the embedding matrix, then do
    all query lookups using row-slices of that matrix — zero extra
    encode calls inside the per-query loop.
    """
    # ← CREATE output_dir FIRST
    indices_dir = output_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    max_k = 100  # Cache always stores top-100 regardless of k_values so the cache can be reused for different k without re-running

    # ── Group pairs by document ────────────────────────────────────────
    doc_positives: dict[str, list[dict]] = defaultdict(list)
    doc_mentions:  dict[str, dict[str, dict]] = defaultdict(dict)

    for pair in pairs:
        doc_id = pair["doc_id"]
        if pair["label"] != "NOREL":
            doc_positives[doc_id].append(pair)

        for mid, trig, sent in [
            ("head_mention_id", "head_trigger", "head_sentence"),
            ("tail_mention_id", "tail_trigger", "tail_sentence"),
        ]:
            if pair[mid] not in doc_mentions[doc_id]:
                doc_mentions[doc_id][pair[mid]] = {
                    "head_mention_id": pair[mid],
                    "head_trigger":    pair[trig],
                    "head_sentence":   pair[sent],
                }

    doc_ids = list(doc_positives.keys())
    print(f"\nEvaluating on {len(doc_ids)} documents...")

    # ── Global accumulators ────────────────────────────────────────────
    total_acc  = {k: {"tp": 0, "retrieved": 0, "total": 0} for k in k_values}
    total_temp = {k: {"tp": 0, "retrieved": 0, "total": 0} for k in k_values}
    total_caus = {k: {"tp": 0, "retrieved": 0, "total": 0} for k in k_values}

    for doc_id in tqdm(doc_ids, desc="Documents"):
        mentions_dict = doc_mentions[doc_id]
        mentions_list = list(mentions_dict.values())
        n = len(mentions_list)
        if n < 2:
            continue

        # ── ONE batched encode call for the whole document ─────────────
        texts = [
            wrap_trigger(m["head_sentence"], m["head_trigger"])
            for m in mentions_list
        ]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Map mention_id -> index in mentions_list (and embeddings)
        mention_idx = {
            m["head_mention_id"]: i
            for i, m in enumerate(mentions_list)
        }

        # Build ground truth for this document
        gt:     dict[str, set] = defaultdict(set)
        gt_cat: dict[str, str] = {}
        for pair in doc_positives[doc_id]:
            hid = pair["head_mention_id"]
            tid = pair["tail_mention_id"]
            gt[hid].add(tid)
            gt_cat[hid] = pair.get("relation_category", "temporal")

        # ── Evaluate using pre-computed embeddings ─────────────────────
        acc, acc_t, acc_c = evaluate_document(
            mentions_list, embeddings, gt, gt_cat, mention_idx,
            k_values, max_k,
        )

        # ── Merge into global accumulators ─────────────────────────────
        for k in k_values:
            for key in ("tp", "retrieved", "total"):
                total_acc[k][key]  += acc.get(k, {}).get(key, 0)
                total_temp[k][key] += acc_t.get(k, {}).get(key, 0)
                total_caus[k][key] += acc_c.get(k, {}).get(key, 0)

        # ← BUILD INDEX for saving + caching
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save embeddings for later use (avoid re-encoding)
        embeddings_path = indices_dir / f"{doc_id}_embeddings.npy"
        np.save(str(embeddings_path), embeddings)

        # Save index + mentions
        index_path = indices_dir / f"{doc_id}.faiss"
        faiss.write_index(index, str(index_path))

        mentions_path = indices_dir / f"{doc_id}_mentions.pkl"
        
        with open(mentions_path, "wb") as f:
            pickle.dump(mentions_list, f)
        
        # ← NEW: Build cache while we have embeddings
        cache_dict = {}
        for q_idx, query_mention in enumerate(mentions_list):
            query_id = query_mention["head_mention_id"]
            query_emb = embeddings[q_idx : q_idx + 1]
            
            # Search for top-100
            _, top_indices = index.search(query_emb, min(100, len(mentions_list)))
            
            wrapped_candidates = []
            for candidate_idx in top_indices[0]:
                candidate = mentions_list[candidate_idx]
                if candidate["head_mention_id"] == query_id:
                    continue  # skip self
                sentence = candidate["head_sentence"]
                trigger = candidate["head_trigger"]
                wrapped_sentence = wrap_trigger(sentence, trigger)  # use wrap_trigger for consistent spacing
                wrapped_candidates.append({
                    "id": candidate["head_mention_id"],
                    "trigger": trigger,
                    "wrapped_sentence": wrapped_sentence,
                })
            cache_dict[query_id] = {
                "query_trigger":  query_mention["head_trigger"],
                "query_sentence": wrap_trigger(query_mention["head_sentence"], query_mention["head_trigger"]),
                "candidates":     wrapped_candidates,
            }
        
        # Save cache
        cache_path = cache_dir / f"{doc_id}_cache.json"
        with open(cache_path, "w") as f:
            json.dump(cache_dict, f, indent=2)

    # ── Compute scores ─────────────────────────────────────────────────
    scores = {}
    for k in k_values:
        p,  r,  f  = prf(total_acc[k]["tp"],  total_acc[k]["retrieved"],  total_acc[k]["total"])
        tp, tr, tf = prf(total_temp[k]["tp"], total_temp[k]["retrieved"], total_temp[k]["total"])
        cp, cr, cf = prf(total_caus[k]["tp"], total_caus[k]["retrieved"], total_caus[k]["total"])
        scores[f"@{k}"] = {
            "precision": p,  "recall": r,  "f1": f,
            "temporal":  {"precision": tp, "recall": tr, "f1": tf},
            "causal":    {"precision": cp, "recall": cr, "f1": cf},
        }

    return scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index and evaluate recall@k for RQ1"
    )
    parser.add_argument("--processed_dir", type=Path, default="data/processed",
                        help="Path to processed pairs")
    parser.add_argument("--output_dir",    type=Path, default="experiments/retriever",
                        help="Path to save FAISS indices, embeddings, and results")
    parser.add_argument("--split", type=str, default="valid",  # ← ADD THIS
                        help="Data split (train/valid/test)")
    parser.add_argument("--k_values",      type=int, nargs="+",
                        default=[5, 10, 20, 30, 50])
    parser.add_argument("--batch_size",    type=int, default=256,
                        help="Sentences per model.encode() batch. "
                             "256 is fast on MPS/CPU; raise to 512 on GPU.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Expects the model to be saved at <output_dir>/model/ by train_retriever.py
    model_path = args.output_dir / "model"
    #model_path = Path(model_path)
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(str(model_path), device=device)

    wem = model._first_module()
    wem.tokenizer.add_tokens([MARKER_OPEN, MARKER_CLOSE], special_tokens=True)
    # Do NOT resize embeddings here — saved model already has correct vocab size.
    
    print("Loading validation pairs...")
    all_pairs = load_jsonl(Path(args.processed_dir) / f"{args.split}_pairs.jsonl")
    n_pos = sum(1 for p in all_pairs if p["label"] != "NOREL")
    print(f"  {len(all_pairs):,} pairs  |  positive: {n_pos:,}  |  NOREL: {len(all_pairs)-n_pos:,}")

    start  = time.time()
    scores = evaluate(
        all_pairs, 
        model, 
        args.k_values,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    elapsed = time.time() - start

    # ── Print results ──────────────────────────────────────────────────
    print("\n=== RQ1 RESULTS ===")
    print(f"{'k':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  "
          f"{'T-P':>7}  {'T-R':>7}  {'T-F1':>7}  "
          f"{'C-P':>7}  {'C-R':>7}  {'C-F1':>7}")
    print("-" * 85)
    for k in args.k_values:
        s = scores[f"@{k}"]
        t = s["temporal"]
        c = s["causal"]
        print(f"@{k:>4}  {s['precision']:>7.4f}  {s['recall']:>7.4f}  {s['f1']:>7.4f}  "
              f"{t['precision']:>7.4f}  {t['recall']:>7.4f}  {t['f1']:>7.4f}  "
              f"{c['precision']:>7.4f}  {c['recall']:>7.4f}  {c['f1']:>7.4f}")

    print(f"\nEvaluation time: {elapsed:.1f}s  "
          f"({elapsed/60:.1f} min)")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)  # ← This creates the dir
    
    with open(out_path / "rq1_results.json", "w") as f:
        json.dump({
            "model_path": str(model_path),
            "k_values":   args.k_values,
            "scores":     scores,
            "time_sec":   round(elapsed, 1),
            "note": (
                "Retriever is BINARY. temporal/causal split shows "
                "coverage by relation type, not model predictions."
            ),
        }, f, indent=2)
    print(f"Results → {out_path / 'rq1_results.json'}")


if __name__ == "__main__":
    main()