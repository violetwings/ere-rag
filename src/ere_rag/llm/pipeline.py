"""
pipeline.py

End-to-end LLM stage for RAG-based Event Relation Extraction:

1. Load cached retriever candidates.
2. For each query event, partition retrieved candidates.
3. Prompt the LLM with a specified-event task.
4. Parse bucketed extraction results into pairwise predictions.
5. Evaluate precision/recall/F1 and save predictions/timing.

This file is intentionally compatible with the cache shape.

Usage:
    python src/ere_rag/llm/pipeline.py \
        --run_dir experiments/retriever/temporal/<run_name> \
        --split valid \
        --n_docs 1 \
        --percentage 0.6 \
        --partition_size 15 \
        --prompt_version v2_detailed_rationale \
        --relation_type temporal \
        --llm_backend api \
        --llm_model "qwen3-next-80b-a3b-instruct" \
        --query_scope all_heads \
        --sample_seed 42

"""

from __future__ import annotations

import argparse
import json
import time
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ere_rag.llm.inference import load_model, run_inference
from ere_rag.llm.parse_output import parse_llm_output, predictions_to_pairs
from ere_rag.llm.prompt import build_prompt


def load_jsonl(filepath: Path) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(filepath: Path, rows: list[dict]) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Path):
        value = str(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:+@%=-]+", text):
        return text
    return json.dumps(text, ensure_ascii=False)

def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def write_yaml(filepath: Path, data: dict) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(_yaml_lines(data)))
        f.write("\n")


def partition_candidates(candidates: list[dict], partition_size: int) -> list[list[dict]]:
    if partition_size <= 0:
        raise ValueError("--partition_size must be positive")
    return [candidates[i : i + partition_size] for i in range(0, len(candidates), partition_size)]

def doc_ids_for_run(pairs_by_doc: dict[str, list[dict]]) -> list[str]:
    return sorted(pairs_by_doc.keys())

def select_candidates(
    all_candidates: list[dict],
    n_doc_mentions: int,
    k: int | None,
    percentage: float | None,
) -> list[dict]:
    """Select top retrieved candidates.

    If percentage is provided, it is computed relative to the other events in
    the document, matching the retriever evaluation convention.
    """
    if percentage is not None:
        cutoff = max(1, round(percentage * max(1, n_doc_mentions - 1)))
    else:
        cutoff = k if k is not None else len(all_candidates)
    return all_candidates[:cutoff]


def document_context_from_cache(cached_entry: dict) -> str | None:
    """Extraction of full document/context text from a cache row.   
       Doesn't return anything for
    """
    for key in ("document", "doc_text", "document_text", "context", "full_text"):
        value = cached_entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("sentences", "document_sentences"):
        value = cached_entry.get(key)
        if isinstance(value, list) and value:
            return " ".join(str(s) for s in value)
    return None


def query_ids_for_document(
    doc_pairs: list[dict],
    cache_dict: dict,
    relation_type: str,
    query_scope: str,
) -> list[str]:
    if query_scope == "all_heads":
        return sorted(cache_dict.keys())

    relevant_pairs = [
        p
        for p in doc_pairs
        if p.get("label") != "NOREL"
        and (relation_type == "both" or p.get("relation_category") == relation_type)
    ]
    return sorted({p["head_mention_id"] for p in relevant_pairs if p.get("head_mention_id") in cache_dict})

def sample_doc_ids(doc_ids: list[str], n_docs: int | None, sample_seed: int) -> list[str]:
    if n_docs is None or n_docs >= len(doc_ids):
        return list(doc_ids)
    rng = random.Random(sample_seed)
    return rng.sample(doc_ids, n_docs)

def evaluate_predictions(predictions: list[dict], pairs: list[dict], relation_type: str) -> dict:
    expected_categories = ["temporal", "causal"] if relation_type == "both" else [relation_type]

    gt = {}
    for pair in pairs:
        category = pair.get("relation_category", "temporal")
        if category not in expected_categories or pair.get("label") == "NOREL":
            continue
        key = (pair["head_mention_id"], pair["tail_mention_id"], category)
        gt[key] = pair["label"]

    pred = {}
    for row in predictions:
        category = row.get("relation_category", "temporal")
        if category not in expected_categories:
            continue
        key = (row["head_mention_id"], row["tail_mention_id"], category)
        pred[key] = row.get("predicted_label", "NOREL")

    def counts(filter_category: str | None = None) -> dict:
        tp = fp = fn = 0
        relevant_gt = {k: v for k, v in gt.items() if filter_category is None or k[2] == filter_category}

        for key, gold_label in relevant_gt.items():
            pred_label = pred.get(key, "NOREL")
            if pred_label == gold_label:
                tp += 1
            elif pred_label == "NOREL":
                fn += 1
            else:
                fp += 1
                fn += 1

        for key, pred_label in pred.items():
            if pred_label == "NOREL":
                continue
            if filter_category is not None and key[2] != filter_category:
                continue
            if key not in gt:
                fp += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    metrics = {"overall": counts()}
    if "temporal" in expected_categories:
        metrics["temporal"] = counts("temporal")
    if "causal" in expected_categories:
        metrics["causal"] = counts("causal")
    return metrics


def run_document(
    doc_pairs: list[dict],
    doc_id: str,
    cache_dict: dict,
    k: int | None,
    percentage: float | None,
    partition_size: int,
    prompt_version: str,
    relation_type: str,
    query_scope: str,
    max_new_tokens: int,
    temperature: float,
    timing_totals: dict,
    verbose: bool,
) -> tuple[list[dict], list[dict]]:
    if not cache_dict:
        return [], []

    n_doc_mentions = len(cache_dict)
    all_predictions: list[dict] = []
    timing_rows: list[dict] = []

    for query_id in query_ids_for_document(doc_pairs, cache_dict, relation_type, query_scope):
        cached_entry = cache_dict.get(query_id)
        if not cached_entry:
            continue

        query_event = {
            "id": query_id,
            "trigger_word": cached_entry.get("query_trigger", cached_entry.get("trigger", "")),
            "wrapped_sentence": cached_entry.get("query_sentence", cached_entry.get("wrapped_sentence", "")),
        }
        candidates = select_candidates(cached_entry.get("candidates", []), n_doc_mentions, k, percentage)
        if not candidates:
            continue

        document_context = document_context_from_cache(cached_entry)
        partitions = partition_candidates(candidates, partition_size)

        for part_idx, partition in enumerate(partitions):
            prompt = build_prompt(
                version=prompt_version,
                query_event=query_event,
                candidates=partition,
                relation_type=relation_type,
                document_context=document_context,
            )

            if verbose:
                print(
                    f"\n[doc={doc_id} query={query_id} partition={part_idx} "
                    f"n_candidates={len(partition)} call={timing_totals.get('llm_n_calls', 0) + 1}]"
                )
                print(prompt)

            response, llm_elapsed = run_inference(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            parsed = parse_llm_output(response, len(partition), partition, relation_type)
            pred_pairs = predictions_to_pairs(query_event, parsed, doc_id)

            all_predictions.extend(pred_pairs)
            timing_totals["llm_total"] = timing_totals.get("llm_total", 0.0) + llm_elapsed
            timing_totals["llm_n_calls"] = timing_totals.get("llm_n_calls", 0) + 1

            timing_rows.append(
                {
                    "doc_id": doc_id,
                    "query_id": query_id,
                    "partition": part_idx,
                    "n_candidates": len(partition),
                    "llm_s": round(llm_elapsed, 3),
                }
            )

            if verbose:
                print("\n--- RAW RESPONSE ---")
                print(response)

    return all_predictions, timing_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ERE classification over cached retrieval")
    parser.add_argument("--run_dir", type=Path, required=True, help="Retriever experiment run directory containing cache/")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"), help="Directory containing <split>_pairs.jsonl")
    parser.add_argument("--split", type=str, default="valid", help="Data split: valid/test/train")
    parser.add_argument("--n_docs", type=int, default=None, help="Limit to the first N eligible documents")
    parser.add_argument("--sample_seed", type=int, default=42, help="Seed for --n_docs random document sampling")

    parser.add_argument("--llm_backend", type=str, default="api", choices=["groq", "ollama", "api", "hf"])
    parser.add_argument("--llm_model", type=str, required=True, help="Model name/id for the selected backend")
    parser.add_argument("--max_new_tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--k", type=int, default=None, help="Use top-k cached candidates per query")
    parser.add_argument("--percentage", type=float, default=None, help="Use top percentage of other document events, e.g. 0.6")
    parser.add_argument("--partition_size", type=int, default=15, help="Number of retrieved candidates per prompt")
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="llmere_few_shot_no_rationale",
        choices=[
            "llmere_compact",
            "llmere_rationale",
            "llmere_few_shot",
            "llmere_few_shot_no_rationale",
            "v1_baseline",
            "v2_detailed_rationale",
            "v3_few_shot",
            "v4_combined",
            "v5_strict",
            "llmere_no_desc_few_shot",
            "llmere_no_desc_rationale",
            "llmere_no_desc",
            "llmere_no_desc_few_shot_no_rationale",
        ],
    )
    parser.add_argument("--relation_type", type=str, default="temporal", choices=["temporal", "causal", "both"])
    parser.add_argument(
        "--query_scope",
        type=str,
        default="positive_heads",
        choices=["positive_heads", "all_heads"],
        help="positive_heads is cheaper; all_heads is the correct choice",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser
def build_run_config(
    args: argparse.Namespace,
    cache_dir: Path,
    output_dir: Path,
    cutoff_tag: str,
    available_doc_ids: list[str],
    selected_doc_ids: list[str],
) -> dict:
    return {
        "run": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "output_dir": str(output_dir),
            "cutoff_tag": cutoff_tag,
        },
        "paths": {
            "run_dir": str(args.run_dir),
            "cache_dir": str(cache_dir),
            "processed_dir": str(args.processed_dir),
        },
        "data": {
            "split": args.split,
            "n_available_docs": len(available_doc_ids),
            "n_selected_docs": len(selected_doc_ids),
            "n_docs_requested": args.n_docs,
            "sample_seed": args.sample_seed,
            "sampling": "all_documents" if args.n_docs is None else "random_sample",
            "selected_doc_ids": selected_doc_ids,
        },
        "retrieval": {
            "k": args.k,
            "percentage": args.percentage,
            "partition_size": args.partition_size,
        },
        "llm": {
            "backend": args.llm_backend,
            "model": args.llm_model,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "prompt": {
            "version": args.prompt_version,
            "relation_type": args.relation_type,
            "query_scope": args.query_scope,
        },
        "runtime": {
            "verbose": args.verbose,
        },
    }

def main() -> None:
    args = build_arg_parser().parse_args()

    if args.k is None and args.percentage is None:
        args.percentage = 0.6

    cache_dir = args.run_dir / "cache"
    run_name = args.run_dir.name
    cutoff_tag = f"pct{args.percentage}" if args.percentage is not None else f"k{args.k}"
    model_tag = args.llm_model.replace("/", "_").replace(":", "_")

    output_dir = (
        Path("experiments")
        / "llm"
        / args.relation_type
        / run_name
        / f"{args.llm_backend}_{model_tag}_nd{args.n_docs}_{cutoff_tag}_split{args.split}_pv{args.prompt_version}_ps{args.partition_size}_{args.query_scope}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    
    pairs_path = args.processed_dir / f"{args.split}_pairs.jsonl"
    print(f"Loading pairs from {pairs_path}")
    all_pairs = load_jsonl(pairs_path)

    pairs_by_doc: dict[str, list[dict]] = defaultdict(list)
    for pair in all_pairs:
        pairs_by_doc[pair["doc_id"]].append(pair)

    available_doc_ids = doc_ids_for_run(pairs_by_doc)
    doc_ids = sample_doc_ids(available_doc_ids, args.n_docs, args.sample_seed)

    config = build_run_config(
        args=args,
        cache_dir=cache_dir,
        output_dir=output_dir,
        cutoff_tag=cutoff_tag,
        available_doc_ids=available_doc_ids,
        selected_doc_ids=doc_ids,
    )
    write_yaml(output_dir / "config.yaml", config)
    print(f"Config saved to -> {output_dir / 'config.yaml'}")


    print(f"Loading LLM backend: {args.llm_backend} (model: {args.llm_model})")
    load_model(backend=args.llm_backend, model_name=args.llm_model, device="cpu")

    print(
        f"Running {len(doc_ids)} docs "
        f"(relation_type={args.relation_type}, prompt={args.prompt_version}, "
        f"partition_size={args.partition_size}, query_scope={args.query_scope}, {cutoff_tag})"
    )

    all_predictions: list[dict] = []
    all_timing: list[dict] = []
    timing_totals: dict = {}
    pipeline_start = time.time()

    for doc_id in tqdm(doc_ids, desc="LLM pipeline"):
        cache_dict = load_cache(cache_dir / f"{doc_id}_cache.json")
        if not cache_dict:
            continue

        predictions, timing_rows = run_document(
            doc_pairs=pairs_by_doc[doc_id],
            doc_id=doc_id,
            cache_dict=cache_dict,
            k=args.k,
            percentage=args.percentage,
            partition_size=args.partition_size,
            prompt_version=args.prompt_version,
            relation_type=args.relation_type,
            query_scope=args.query_scope,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timing_totals=timing_totals,
            verbose=args.verbose,
        )
        all_predictions.extend(predictions)
        all_timing.extend(timing_rows)

    pipeline_elapsed = time.time() - pipeline_start
    evaluated_pairs = [p for p in all_pairs if p["doc_id"] in set(doc_ids)]
    metrics = evaluate_predictions(all_predictions, evaluated_pairs, args.relation_type)

    write_jsonl(output_dir / "predictions.jsonl", all_predictions)
    write_jsonl(output_dir / "timing.jsonl", all_timing)

    results = {
        "cache_dir": str(cache_dir),
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
        "prompt_version": args.prompt_version,
        "relation_type": args.relation_type,
        "query_scope": args.query_scope,
        "k": args.k,
        "percentage": args.percentage,
        "partition_size": args.partition_size,
        "n_docs": len(doc_ids),
        "n_predictions": len(all_predictions),
        "metrics": metrics,
        "timing": {
            "total_s": round(pipeline_elapsed, 1),
            "per_doc_s": round(pipeline_elapsed / max(1, len(doc_ids)), 2),
            "llm_n_calls": timing_totals.get("llm_n_calls", 0),
            "avg_llm_call_s": round(
                timing_totals.get("llm_total", 0.0) / max(1, timing_totals.get("llm_n_calls", 0)),
                2,
            ),
        },
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== PIPELINE RESULTS ===")
    print(f"Documents processed: {len(doc_ids)}")
    print(f"Total predictions:   {len(all_predictions)}")
    print(f"LLM calls:           {timing_totals.get('llm_n_calls', 0)}")
    print(f"Total wall time:     {pipeline_elapsed:.1f}s")
    print("\n           P        R        F1")
    for category in ("overall", "temporal", "causal"):
        if category in metrics:
            row = metrics[category]
            print(f"  {category:<10} {row['precision']:.4f}  {row['recall']:.4f}  {row['f1']:.4f}")
    print(f"\nResults saved to -> {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
