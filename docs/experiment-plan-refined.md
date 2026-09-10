# Experiment Plan

This document records every design decision for the refined experimental stage,
with justifications. For detailed run commands, see `src/ere_rag/eci/README.md`.

---

## Stage 1 — Retriever
No changes were made on this stage.
---

## Stage 2 — LLM Experiments

### Common settings (fixed across all LLM runs)

| Parameter | Value |
|---|---|
| Split | valid |
| n_docs | **20** (compute budget) |
| Retriever model | seed=42 for only the causal category |
| llm_backend | api |
| llm_model | qwen3-next-80b-a3b-instruct |
| temperature | 0.0 (deterministic) |
| max_new_tokens | Depends on the run mode |

### RQ2 and RQ3 configurations (4 per relation type)

RQ2 compares A vs B vs C (precision and call count/time).
RQ3 compares B vs D (precision; B doubles as the rationale-OFF arm).

| Config | Label | % cutoff | Mode | Purpose |
|---|---|---|---|---|---|
| A | per-query | - |`per_query_no_retriever` | RQ2: per-query calls without filtering |
| B | per-query + filtered | 50%| `per_query_retriever` | RQ2: per-query calls filtered |
| C | whole document | - | `whole_doc_no_reasoning` | RQ2: per-document calls baseline |
| D | whole document + reasoning | - | `whole_doc_reasoning` | RQ3: reasoning-ON arm (also the full proposed pipeline / headline "ours") |

Config C is reported in both RQ2 and RQ3 tables — the same run, both tables, not two separate runs.

### Causal experiment commands

```bash
CAUSAL_DIR=experiments/retriever/causal/grouped_causal_k5_seed42_ep5_bs16_mg5000_cpu

# Config A — per-query
python src/ere_rag/eci/pipeline.py \
    --raw_data_path data/raw/MAVEN_ERE/valid.jsonl \
    --processed_dir data/processed \
    --run_dir $CAUSAL_DIR \
    --split valid \
    --n_docs 20 \
    --mode per_query_no_retriever \
    --llm_backend api \
    --llm_model qwen3-next-80b-a3b-instruct

# Config B — per-query + filtered
python src/ere_rag/eci/pipeline.py \
        --raw_data_path data/raw/MAVEN_ERE/valid.jsonl \
        --processed_dir data/processed \
        --run_dir $CAUSAL_DIR \
        --split valid \
        --n_docs 20 \
        --percentage 0.5 \
        --mode per_query_retriever \
        --llm_backend api \
        --llm_model qwen3-next-80b-a3b-instruct

# Config C — whole document
python src/ere_rag/eci/pipeline.py \
    --raw_data_path data/raw/MAVEN_ERE/valid.jsonl \
    --processed_dir data/processed \
    --run_dir $CAUSAL_DIR \
    --split valid \
    --n_docs 20 \
    --mode whole_doc_no_reasoning \
    --llm_backend api \
    --llm_model qwen3-next-80b-a3b-instruct

# Config D — whole document + reasoning
python src/ere_rag/eci/pipeline.py \
    --raw_data_path data/raw/MAVEN_ERE/valid.jsonl \
    --processed_dir data/processed \
    --run_dir $CAUSAL_DIR \
    --split valid \
    --n_docs 20 \
    --mode whole_doc_reasoning \
    --llm_backend api \
    --llm_model qwen3-next-80b-a3b-instruct
```

Results and analysis: `experiments/eci/README.md`.

---
