"""
pipeline.py

Binary Event Causality Identification (ECI) over MAVEN-ERE. No CAUSE vs
PRECONDITION distinction, no temporal relations, no legacy prompt_version
list -- exactly four modes:

  whole_doc_reasoning     one call per document, with a REASONING section
  whole_doc_no_reasoning  one call per document, straight to the answer
  per_query_no_retriever  one call per query event, candidates = every
                          other event mention in the document
  per_query_retriever     one call per query event, candidates = the
                          retriever's top-cutoff ranked list for that
                          query (needs --run_dir for the retriever's cache)

Document text always comes from document_text.py (raw MAVEN-ERE data), not
from the retriever cache -- the cache (when used) only supplies a ranked
candidate ORDER for per_query_retriever, never display text.

Usage:
    python src/ere_rag/eci/pipeline.py \
        --raw_data_path data/raw/MAVEN_ERE/valid.jsonl \
        --processed_dir data/processed \
        --split valid \
        --n_docs 20 \
        --mode whole_doc_reasoning \
        --llm_backend api \
        --llm_model qwen3-next-80b-a3b-instruct

    python src/ere_rag/eci/pipeline.py \
        --raw_data_path data/raw/MAVEN_ERE/valid.jsonl \
        --processed_dir data/processed \
        --run_dir experiments/retriever/causal/grouped_causal_k5_seed42_ep5_bs16_mg5000_cpu \
        --split valid \
        --n_docs 1 \
        --percentage 0.5 \
        --mode per_query_retriever \
        --llm_backend api \
        --llm_model qwen3-next-80b-a3b-instruct
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import ere_rag.eci.document_text as document_text
import ere_rag.eci.parse_output as parse_output
import ere_rag.eci.prompt as prompt_mod
from ere_rag.eci.inference import load_model, run_inference_messages

MODES = ["whole_doc_reasoning", "whole_doc_no_reasoning", "per_query_no_retriever", "per_query_retriever"]

# Per-mode default token budgets, used only when --max_new_tokens isn't
# passed. Whole-document modes need headroom for many implicit pairs (and
# a reasoning preamble, for the reasoning config); per-query modes answer
# one query against a bounded candidate list with no reasoning, so they
# can be much cheaper.
DEFAULT_MAX_NEW_TOKENS = {
    "whole_doc_reasoning": 8000,
    "whole_doc_no_reasoning": 4000,
    "per_query_no_retriever": 300,
    "per_query_retriever": 300,
}


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def load_jsonl(filepath: Path) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(filepath: Path, rows: list[dict]) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Candidate selection (retriever cutoff, unchanged convention: percentage is
# relative to the OTHER events in the document)
# --------------------------------------------------------------------------- #

def compute_cutoff(n_context_events: int, percentage: float | None, k: int | None, n_available: int) -> int:
    """Single source of truth for the cutoff arithmetic, exposed on its own
    so logging can print the exact number applied without recomputing it
    separately (which risks the printed value silently drifting from the
    real one if either copy is edited later).
    """
    if percentage is not None:
        return max(1, round(percentage * max(1, n_context_events - 1)))
    return k if k is not None else n_available


def select_top(ranked_ids: list[str], n_context_events: int, percentage: float | None, k: int | None) -> list[str]:
    cutoff = compute_cutoff(n_context_events, percentage, k, len(ranked_ids))
    return ranked_ids[:cutoff]


# --------------------------------------------------------------------------- #
# Whole-document mode
# --------------------------------------------------------------------------- #

def run_whole_document(
    doc_id: str,
    reasoning: bool,
    max_new_tokens: int,
    temperature: float,
    timing_totals: dict,
    verbose: bool,
    reasoning_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    event_ids = document_text.event_ids_for_doc(doc_id)
    if not event_ids:
        return [], []

    marker_for_mention = {eid: f"e{i}" for i, eid in enumerate(event_ids)}
    id_map = {label: eid for eid, label in marker_for_mention.items()}
    marked_document = document_text.build_marked_text(doc_id, marker_for_mention, full_document=True)
    if marked_document is None:
        return [], []

    messages = prompt_mod.build_whole_document_messages(marked_document, reasoning=reasoning)
    if verbose:
        print(f"\n[doc={doc_id} WHOLE-DOCUMENT n_events={len(event_ids)} reasoning={reasoning} "
              f"call={timing_totals.get('llm_n_calls', 0) + 1}]")
        for m in messages:
            print(f"--- {m['role'].upper()} ---")
            print(m["content"])

    response, elapsed = run_inference_messages(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    timing_totals["llm_total"] = timing_totals.get("llm_total", 0.0) + elapsed
    timing_totals["llm_n_calls"] = timing_totals.get("llm_n_calls", 0) + 1
    if verbose:
        print("\n--- RAW RESPONSE ---")
        print(response)

    raw_pairs = parse_output.parse_pairs(response, valid_labels=set(id_map.keys()))
    predictions = [
        {"doc_id": doc_id, "head_mention_id": id_map[p["head"]], "tail_mention_id": id_map[p["tail"]], "rationale": p["rationale"]}
        for p in raw_pairs
    ]

    if reasoning:
        preamble = parse_output.extract_reasoning_preamble(response)
        if preamble:
            reasoning_rows.append({"doc_id": doc_id, "reasoning": preamble})

    timing_rows = [{"doc_id": doc_id, "n_events": len(event_ids), "n_predictions": len(predictions), "llm_s": round(elapsed, 3)}]
    return predictions, timing_rows


# --------------------------------------------------------------------------- #
# Per-query mode (no-retriever and retriever-filtered share this function --
# they only differ in how candidate_ids_for_query is computed by the caller)
# --------------------------------------------------------------------------- #

def run_per_query(
    doc_id: str,
    use_retriever: bool,
    cache_dir: Path | None,
    percentage: float | None,
    k: int | None,
    max_new_tokens: int,
    temperature: float,
    timing_totals: dict,
    verbose: bool,
) -> tuple[list[dict], list[dict]]:
    print(f"[percentage-received] run_per_query got percentage={percentage!r} (type={type(percentage).__name__}), k={k!r}")
    event_ids = document_text.event_ids_for_doc(doc_id)
    if not event_ids:
        return [], []
    event_ids_set = set(event_ids)

    cache_dict: dict = {}
    if use_retriever:
        cache_dict = load_cache(cache_dir / f"{doc_id}_cache.json")
        if not cache_dict:
            return [], []

    predictions: list[dict] = []
    timing_rows: list[dict] = []

    for query_id in event_ids:
        if use_retriever:
            cached_entry = cache_dict.get(query_id)
            if not cached_entry:
                continue
            ranked = [c.get("id") for c in cached_entry.get("candidates", []) if c.get("id") in event_ids_set]
            cutoff = compute_cutoff(len(event_ids), percentage, k, len(ranked))
            top = select_top(ranked, len(event_ids), percentage, k)
            # Retriever rank decides WHICH events are included; document
            # reading order decides how they're labeled e0, e1, e2... --
            # otherwise the marker numbering follows similarity rank and
            # bears no relation to where events actually appear in the text.
            candidate_ids = document_text.sort_by_reading_order(doc_id, top)
            if verbose:
                print(f"  [cutoff] doc={doc_id} query={query_id}: cache_raw={len(cached_entry.get('candidates', []))} "
                      f"-> id_matched={len(ranked)} -> computed_cutoff={cutoff} -> kept={len(candidate_ids)} "
                      f"(percentage={percentage!r}, k={k!r}, n_context_events={len(event_ids)})")
        else:
            candidate_ids = [eid for eid in event_ids if eid != query_id]

        if not candidate_ids:
            continue

        marker_for_mention = {query_id: "q"}
        id_map = {"q": query_id}
        for i, cid in enumerate(candidate_ids):
            marker_for_mention[cid] = f"e{i}"
            id_map[f"e{i}"] = cid

        snippet = document_text.build_marked_text(doc_id, marker_for_mention, full_document=False)
        if snippet is None:
            continue

        candidate_labels = [f"e{i}" for i in range(len(candidate_ids))]
        messages = prompt_mod.build_per_query_messages(snippet, candidate_labels)
        if verbose:
            print(f"\n[doc={doc_id} query={query_id} n_candidates={len(candidate_ids)} "
                  f"retriever={use_retriever} call={timing_totals.get('llm_n_calls', 0) + 1}]")
            for m in messages:
                print(f"--- {m['role'].upper()} ---")
                print(m["content"])

        response, elapsed = run_inference_messages(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        timing_totals["llm_total"] = timing_totals.get("llm_total", 0.0) + elapsed
        timing_totals["llm_n_calls"] = timing_totals.get("llm_n_calls", 0) + 1
        if verbose:
            print("\n--- RAW RESPONSE ---")
            print(response)

        raw_pairs = parse_output.parse_pairs(response, valid_labels=set(id_map.keys()))
        n_reversed = sum(1 for p in raw_pairs if p["head"] != "q")
        if n_reversed and verbose:
            print(f"  [NOTE] doc={doc_id} query={query_id}: dropped {n_reversed} pair(s) with candidate as "
                  f"head -- reverse direction not requested, likely instruction-following slip")
        raw_pairs = [p for p in raw_pairs if p["head"] == "q"]
        for p in raw_pairs:
            predictions.append({
                "doc_id": doc_id,
                "head_mention_id": id_map[p["head"]],
                "tail_mention_id": id_map[p["tail"]],
                "rationale": "",
            })

        timing_rows.append({
            "doc_id": doc_id, "query_id": query_id, "n_candidates": len(candidate_ids),
            "n_retrieved_before_cutoff": len(ranked) if use_retriever else None,
            "n_reversed_dropped": n_reversed, "llm_s": round(elapsed, 3),
        })

    return predictions, timing_rows


# --------------------------------------------------------------------------- #
# Evaluation -- binary RELATED/NOREL, causal relations only
# --------------------------------------------------------------------------- #

def evaluate_predictions(predictions: list[dict], gold_pairs: list[dict]) -> dict:
    gt = {
        (p["head_mention_id"], p["tail_mention_id"])
        for p in gold_pairs
        if p.get("relation_category") == "causal" and p.get("label") != "NOREL"
    }
    pred = {(r["head_mention_id"], r["tail_mention_id"]) for r in predictions}

    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary ECI over MAVEN-ERE -- four call-shape/retriever configs")
    parser.add_argument("--raw_data_path", type=Path, required=True, help="Raw MAVEN-ERE jsonl for --split (has 'events'/'tokens' keys)")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"), help="Directory containing <split>_pairs.jsonl (gold labels for evaluation)")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--n_docs", type=int, default=None, help="Limit to a random sample of N eligible documents")
    parser.add_argument("--sample_seed", type=int, default=42)

    parser.add_argument("--mode", type=str, required=True, choices=MODES)
    parser.add_argument("--run_dir", type=Path, default=None, help="Retriever run directory containing cache/ -- required for --mode per_query_retriever")
    parser.add_argument("--percentage", type=float, default=None, help="Retriever cutoff as a fraction of other document events, e.g. 0.5 -- only used by per_query_retriever")
    parser.add_argument("--k", type=int, default=None, help="Retriever cutoff as a fixed count -- alternative to --percentage, only used by per_query_retriever")

    parser.add_argument("--llm_backend", type=str, default="api", choices=["api", "groq", "ollama"])
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Overrides the per-mode default in DEFAULT_MAX_NEW_TOKENS if given")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode == "per_query_retriever" and args.run_dir is None:
        raise SystemExit("--mode per_query_retriever requires --run_dir (the retriever run directory containing cache/)")
    if args.mode == "per_query_no_retriever" and (args.percentage is not None or args.k is not None):
        print(
            f"[WARNING] --mode per_query_no_retriever ignores --percentage/--k entirely -- every query event "
            f"gets ALL other document events as candidates, regardless of what you passed. If you meant to "
            f"filter candidates by retriever similarity, use --mode per_query_retriever instead."
        )
    if args.mode == "per_query_retriever" and args.percentage is None and args.k is None:
        print("[NOTE] neither --percentage nor --k given for per_query_retriever -- defaulting to --percentage 0.5")
        args.percentage = 0.5

    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.mode]

    print(f"Loading raw MAVEN-ERE data from {args.raw_data_path}")
    document_text.load_raw_maven_ere(args.raw_data_path)
    n_mentions = sum(len(document_text.event_ids_for_doc(d)) for d in document_text._RAW_DOC_INDEX)
    print(f"Indexed {len(document_text._RAW_DOC_INDEX)} documents / {n_mentions} event mentions")

    pairs_path = args.processed_dir / f"{args.split}_pairs.jsonl"
    print(f"Loading gold pairs from {pairs_path}")
    all_pairs = load_jsonl(pairs_path)
    pairs_by_doc: dict[str, list[dict]] = defaultdict(list)
    for pair in all_pairs:
        pairs_by_doc[pair["doc_id"]].append(pair)

    cache_dir = (args.run_dir / "cache") if args.run_dir else None

    # Sample from the SAME population, with the SAME seed, regardless of
    # mode -- random.Random(seed).sample() depends on the exact contents
    # and order of the population it's given, so filtering the population
    # down before sampling (e.g. to cache-available docs only) changes
    # which 20 documents get drawn even with an identical seed. Any
    # per-mode requirement (like retriever cache availability) is checked
    # AFTER sampling instead, so every mode evaluates on the same document
    # set whenever that requirement happens to be satisfied for all of
    # them -- which it should be, since the cache is built for the full
    # available pool, not a mode-specific subset.
    available_doc_ids = sorted(set(document_text._RAW_DOC_INDEX.keys()) & set(pairs_by_doc.keys()))
    if not available_doc_ids:
        raise SystemExit("No documents available for this configuration -- check --raw_data_path / --processed_dir / --run_dir all point at the same split.")

    if args.n_docs is not None and args.n_docs < len(available_doc_ids):
        rng = random.Random(args.sample_seed)
        doc_ids = rng.sample(available_doc_ids, args.n_docs)
    else:
        doc_ids = available_doc_ids

    if args.mode == "per_query_retriever":
        missing_cache = [d for d in doc_ids if not (cache_dir / f"{d}_cache.json").exists()]
        if missing_cache:
            print(f"[WARNING] {len(missing_cache)} of the {len(doc_ids)} sampled documents have no retriever "
                  f"cache and will be skipped for --mode per_query_retriever, so this run will NOT be on the "
                  f"same document set as the other three modes: {missing_cache}")
            doc_ids = [d for d in doc_ids if d not in missing_cache]
        if not doc_ids:
            raise SystemExit("None of the sampled documents have a retriever cache -- check --run_dir.")

    output_dir = Path("experiments") / "eci" / args.mode / f"{args.llm_backend}_{args.llm_model.replace('/', '_')}_nd{len(doc_ids)}_split{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LLM backend: {args.llm_backend} (model: {args.llm_model})")
    load_model(backend=args.llm_backend, model_name=args.llm_model)

    print(f"Running {len(doc_ids)} docs (mode={args.mode}, max_new_tokens={max_new_tokens})")

    all_predictions: list[dict] = []
    all_timing: list[dict] = []
    all_reasoning: list[dict] = []
    timing_totals: dict = {}
    start = time.time()

    for doc_id in tqdm(doc_ids, desc="ECI pipeline"):
        if args.mode == "whole_doc_reasoning":
            predictions, timing_rows = run_whole_document(
                doc_id, reasoning=True, max_new_tokens=max_new_tokens, temperature=args.temperature,
                timing_totals=timing_totals, verbose=args.verbose, reasoning_rows=all_reasoning,
            )
        elif args.mode == "whole_doc_no_reasoning":
            predictions, timing_rows = run_whole_document(
                doc_id, reasoning=False, max_new_tokens=max_new_tokens, temperature=args.temperature,
                timing_totals=timing_totals, verbose=args.verbose, reasoning_rows=all_reasoning,
            )
        elif args.mode == "per_query_no_retriever":
            predictions, timing_rows = run_per_query(
                doc_id, use_retriever=False, cache_dir=None, percentage=None, k=None,
                max_new_tokens=max_new_tokens, temperature=args.temperature,
                timing_totals=timing_totals, verbose=args.verbose,
            )
        else:  # per_query_retriever
            predictions, timing_rows = run_per_query(
                doc_id, use_retriever=True, cache_dir=cache_dir, percentage=args.percentage, k=args.k,
                max_new_tokens=max_new_tokens, temperature=args.temperature,
                timing_totals=timing_totals, verbose=args.verbose,
            )
        all_predictions.extend(predictions)
        all_timing.extend(timing_rows)

    elapsed_total = time.time() - start
    evaluated_pairs = [p for p in all_pairs if p["doc_id"] in set(doc_ids)]
    metrics = evaluate_predictions(all_predictions, evaluated_pairs)

    write_jsonl(output_dir / "predictions.jsonl", all_predictions)
    write_jsonl(output_dir / "timing.jsonl", all_timing)
    if all_reasoning:
        write_jsonl(output_dir / "reasoning.jsonl", all_reasoning)

    results = {
        "mode": args.mode,
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
        "percentage": args.percentage,
        "k": args.k,
        "n_docs": len(doc_ids),
        "n_predictions": len(all_predictions),
        "metrics": metrics,
        "timing": {
            "total_s": round(elapsed_total, 1),
            "per_doc_s": round(elapsed_total / max(1, len(doc_ids)), 2),
            "llm_n_calls": timing_totals.get("llm_n_calls", 0),
            "avg_llm_call_s": round(timing_totals.get("llm_total", 0.0) / max(1, timing_totals.get("llm_n_calls", 0)), 2),
        },
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== RESULTS ===")
    print(f"Mode:              {args.mode}")
    print(f"Documents:         {len(doc_ids)}")
    print(f"Predictions:       {len(all_predictions)}")
    print(f"LLM calls:         {timing_totals.get('llm_n_calls', 0)}")
    print(f"Total wall time:   {elapsed_total:.1f}s")
    print(f"Precision/Recall/F1: {metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}  (tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})")
    print(f"\nSaved to -> {output_dir}")


if __name__ == "__main__":
    main()