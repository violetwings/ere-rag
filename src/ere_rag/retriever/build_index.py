"""
build_index.py

Encodes all event mentions for one category (temporal/causal/both) using
the fine-tuned S-BERT model and evaluates retriever recall@P% for RQ1,
where P% is a PERCENTAGE OF A DOCUMENT'S OTHER EVENTS rather than a fixed
k — so the retrieval budget scales with document size instead of being a
flat count that's too small for big documents and too large for small ones.

For a document with n total candidate mentions, percentage p retrieves:
    pct_k = max(1, round(p * (n - 1)))


Performance design:
  - All mentions in a document are encoded in ONE batched model.encode()
    call. The resulting embedding matrix is stored and reused for both
    index construction and all query lookups.
  - FAISS search runs ONCE per query
    per percentage — eliminates redundant searches.
  - No per-query model.encode() calls inside any loop.
  - Cache (top-100% per query) , so
    pipeline.py's downstream loading is untouched — only the RQ1
    EVALUATION cutoff is percentage-based, not the cache depth.

Usage:
    python src/ere_rag/retriever/build_index.py \
        --category temporal \
        --output_dir experiments/retriever/temporal/<run_name> \
        --processed_dir data/processed \
        --percentages 5 10 20 30 50 60
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

TEMPORAL_RELATIONS = [
    "BEFORE", "OVERLAP", "CONTAINS",
    "SIMULTANEOUS", "ENDS-ON", "BEGINS-ON"
]
CAUSAL_RELATIONS = ["CAUSE", "PRECONDITION"]

NEGATIVE_LABEL  = "NOREL"
VALID_CATEGORIES = ("temporal", "causal", "both")

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


def load_pairs_for_category(processed_dir: Path, split: str, category: str) -> list[dict]:
    """
    temporal/causal: load that category's own pairs file.
    both: load AND MERGE both files — safe because positivity is decided
    per-row by is_positive_label(), not by which file the row came from.
    """
    if category in ("temporal", "causal"):
        path = processed_dir / f"{split}_{category}_pairs.jsonl"
        return load_jsonl(path)
    if category == "both":
        t_path = processed_dir / f"{split}_temporal_pairs.jsonl"
        c_path = processed_dir / f"{split}_causal_pairs.jsonl"
        return load_jsonl(t_path) + load_jsonl(c_path)
    raise ValueError(f"Unknown category: {category!r}")


def is_positive_label(label: str, category: str) -> bool:
    if category == "temporal":
        return label in TEMPORAL_RELATIONS
    if category == "causal":
        return label in CAUSAL_RELATIONS
    if category == "both":
        return label in TEMPORAL_RELATIONS or label in CAUSAL_RELATIONS
    raise ValueError(f"Unknown category: {category!r}")


# ── Metrics ─────────────────────────────────────────────────────────────────

def prf(tp: int, retrieved: int, relevant: int) -> tuple[float, float, float]:
    p  = tp / retrieved if retrieved > 0 else 0.0
    r  = tp / relevant  if relevant  > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ── Per-document evaluation ──────────────────────────────────────────────────

def evaluate_document(
    mentions:    list[dict],
    embeddings:  np.ndarray,
    gt:          dict[str, set],
    gt_cat:      dict[str, str],
    mention_idx: dict[str, int],
    percentages: list[float],
) -> tuple[dict, dict, dict, int]:
    """
    Evaluate one document using pre-computed embeddings, at each
    percentage cutoff (relative to n-1, this document's OTHER events).

    Runs FAISS search once per query
    percentage — no extra search needed per percentage.

    Returns accumulators for overall, temporal, causal (each keyed by
    the float percentage, e.g. 0.2), plus the number of queries
    evaluated in this document (for the avg-retrieved diagnostic).
    """
    n = len(mentions)
    if n < 2:
        return {}, {}, {}, 0

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    acc          = {p: {"tp": 0, "retrieved": 0, "total": 0} for p in percentages}
    n_queries    = 0

    for hid, true_tails in gt.items():
        q_idx = mention_idx.get(hid)
        if q_idx is None:
            continue
        n_queries += 1

        query_emb = embeddings[q_idx : q_idx + 1]
        cat       = gt_cat.get(hid, "temporal")

        
        _, raw_indices = index.search(query_emb, n)

        retrieved_all: list[str] = []
        for idx in raw_indices[0]:
            cand_id = mentions[idx]["head_mention_id"]
            if cand_id != hid:
                retrieved_all.append(cand_id)
            

        for pct in percentages:
            # Cutoff relative to n-1 (this doc's other events)
            pct_k = max(1, round(pct * (n - 1)))
            pct_k = min(pct_k, len(retrieved_all))
            retrieved_ids = set(retrieved_all[:pct_k])
            tp = len(true_tails & retrieved_ids)

            acc[pct]["tp"]        += tp
            acc[pct]["retrieved"] += len(retrieved_ids)
            acc[pct]["total"]     += len(true_tails)


    return acc, n_queries


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(
    pairs:      list[dict],
    category:   str,
    model:      SentenceTransformer,
    percentages: list[float],
    output_dir: Path,
    batch_size: int = 256,
) -> dict:
    """
    Evaluate retriever recall@P% across all documents.

    Same per-document batched-encode optimisation as before — only the
    evaluation cutoff (percentage vs fixed k) and the category-aware
    positive filtering are new.
    """
    indices_dir = output_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)


    # ── Group pairs by document ────────────────────────────────────────
    doc_positives: dict[str, list[dict]] = defaultdict(list)
    doc_mentions:  dict[str, dict[str, dict]] = defaultdict(dict)

    for pair in pairs:
        doc_id = pair["doc_id"]
        if is_positive_label(pair["label"], category):
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

    doc_ids = list(doc_mentions.keys())
    print(f"\nEvaluating on {len(doc_ids)} documents (category={category})...")

    # ── Global accumulators ────────────────────────────────────────────
    total_acc    = {p: {"tp": 0, "retrieved": 0, "total": 0} for p in percentages}
    total_queries = 0

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

        mention_idx = {
            m["head_mention_id"]: i
            for i, m in enumerate(mentions_list)
        }

      
        gt:     dict[str, set] = defaultdict(set)
        gt_cat: dict[str, str] = {}
        for pair in doc_positives[doc_id]:
            hid = pair["head_mention_id"]
            tid = pair["tail_mention_id"]
            gt[hid].add(tid)
            gt_cat[hid] = pair.get("relation_category", "temporal") if category == "both" else category

        # ── Evaluate using pre-computed embeddings ─────────────────────
        acc, n_q = evaluate_document(
            mentions_list, embeddings, gt, gt_cat, mention_idx,
            percentages,
        )
        total_queries += n_q

        # ── Merge into global accumulators ─────────────────────────────
        for p in percentages:
            for key in ("tp", "retrieved", "total"):
                total_acc[p][key] += acc.get(p, {}).get(key, 0)

        # ── Build index for saving + caching (unchanged) ───────────────
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        embeddings_path = indices_dir / f"{doc_id}_embeddings.npy"
        np.save(str(embeddings_path), embeddings)

        index_path = indices_dir / f"{doc_id}.faiss"
        faiss.write_index(index, str(index_path))

        mentions_path = indices_dir / f"{doc_id}_mentions.pkl"
        with open(mentions_path, "wb") as f:
            pickle.dump(mentions_list, f)

        # ── Cache (top-100% per query) ──────
        cache_dict = {}
        for q_idx, query_mention in enumerate(mentions_list):
            query_id  = query_mention["head_mention_id"]
            query_emb = embeddings[q_idx : q_idx + 1]

            _, top_indices = index.search(query_emb, len(mentions_list))

            wrapped_candidates = []
            for candidate_idx in top_indices[0]:
                candidate = mentions_list[candidate_idx]
                if candidate["head_mention_id"] == query_id:
                    continue
                sentence = candidate["head_sentence"]
                trigger  = candidate["head_trigger"]
                wrapped_sentence = wrap_trigger(sentence, trigger)
                wrapped_candidates.append({
                    "id":               candidate["head_mention_id"],
                    "trigger":          trigger,
                    "wrapped_sentence": wrapped_sentence,
                })
            cache_dict[query_id] = {
                "query_trigger":  query_mention["head_trigger"],
                "query_sentence": wrap_trigger(query_mention["head_sentence"], query_mention["head_trigger"]),
                "candidates":     wrapped_candidates,
            }

        cache_path = cache_dir / f"{doc_id}_cache.json"
        with open(cache_path, "w") as f:
            json.dump(cache_dict, f, indent=2)

    # ── Compute scores ─────────────────────────────────────────────────
    scores = {}
    for p in percentages:
        prec, rec, f1 = prf(
            total_acc[p]["tp"], total_acc[p]["retrieved"], total_acc[p]["total"]
        )
        avg_retrieved = total_acc[p]["retrieved"] / max(1, total_queries)
        label = f"{round(p*100)}%"
        scores[label] = {
            "precision":     prec,
            "recall":        rec,
            "f1":            f1,
            "avg_retrieved": round(avg_retrieved, 1),
        }

    return scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index and evaluate recall@P% for RQ1"
    )
    parser.add_argument(
        "--category", type=str, required=True, choices=VALID_CATEGORIES,
        help="Which relation type(s) to evaluate against."
    )
    parser.add_argument("--processed_dir", type=Path, default="data/processed",
                        help="Path to processed pairs")
    parser.add_argument("--output_dir",    type=Path, required=True,
                        help="Run folder containing model/ — also where "
                             "indices/, cache/, and rq1_results.json go")
    parser.add_argument("--split", type=str, default="valid",
                        help="Data split (train/valid/test)")
    parser.add_argument(
        "--percentages", type=int, nargs="+", default=[10, 20, 30, 50],
        help="Retrieval cutoffs as percentages of a document's OTHER "
             "events (n-1), e.g. '10 20 30 50' means 10%%, 20%%, 30%%, "
             "50%%. Defaults bracket the ~0.25 average relation density "
    )
    parser.add_argument("--batch_size",    type=int, default=256,
                        help="Sentences per model.encode() batch. "
                             "256 is fast on MPS/CPU; raise to 512 on GPU.")
    args = parser.parse_args()

    percentages = [p / 100 for p in args.percentages]

    device = get_device()
    print(f"Device: {device}")

    model_path = args.output_dir / "model"
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(str(model_path), device=device)

    wem = model._first_module()
    wem.tokenizer.add_tokens([MARKER_OPEN, MARKER_CLOSE], special_tokens=True)
    

    print(f"Loading {args.split} pairs for category={args.category}...")
    all_pairs = load_pairs_for_category(args.processed_dir, args.split, args.category)
    n_pos = sum(1 for p in all_pairs if is_positive_label(p["label"], args.category))
    print(f"  {len(all_pairs):,} pairs  |  positive: {n_pos:,}  |  other: {len(all_pairs)-n_pos:,}")

    start  = time.time()
    scores = evaluate(
        all_pairs,
        args.category,
        model,
        percentages,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start

    # ── Print results ──────────────────────────────────────────────────
    print(f"\n=== RQ1 RESULTS (category={args.category}) ===")
    print(f"{'%':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'avg_k':>7}")
    print("-" * 85)
    for p in args.percentages:
        label = f"{p}%"
        s = scores[label]
        print(f"{label:>5}  {s['precision']:>7.4f}  {s['recall']:>7.4f}  "
              f"{s['f1']:>7.4f}  {s['avg_retrieved']:>7.1f}")

    print(f"\nEvaluation time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "rq1_results.json", "w") as f:
        json.dump({
            "category":    args.category,
            "model_path":  str(model_path),
            "percentages": args.percentages,
            "scores":      scores,
            "time_sec":    round(elapsed, 1),
        }, f, indent=2)
    print(f"Results → {out_path / 'rq1_results.json'}")


if __name__ == "__main__":
    main()