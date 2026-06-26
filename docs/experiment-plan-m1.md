# Experiment Plan

This document records every design decision for each experimental stage,
with justifications. For k-value justification, see
`src/ere_rag/data/stats_README.md`. For full methodology, see
`docs/methodology.md`. And for detailed run commands, `see src/ere_rag/retriever/README.md` and `src/ere_rag/llm/README.md`.

---

## Stage 1 — Retriever

### Data preparation
```bash
# Always seed=42 for data prep — only training seed varies
python src/ere_rag/retriever/prepare_data.py \
    --category temporal --data_dir data/raw/MAVEN_ERE \
    --output_dir data/interim/temporal --sample_rate 0.15 --seed 42

python src/ere_rag/retriever/prepare_data.py \
    --category causal --data_dir data/raw/MAVEN_ERE \
    --output_dir data/interim/causal --sample_rate 0.15 --seed 42
```

**Note**: 15% train-subset sampling seed (42) is fixed and separate from
the training seed. This means all three training seeds see exactly the
same training data — variance across seeds comes from model initialisation
and negative sampling, not data selection.

### Training configuration

| Parameter | Temporal | Causal | Justification |
|---|---|---|---|
| Base model | `all-MiniLM-L6-v2` | same | Best S-BERT config in Efficient ERE [2] on MAVEN-ERE |
| k_negatives | **3** | **5** | See `src/ere_rag/data/stats_README.md` |
| max_groups | **10,000** | **5,000** | Compute budget; causal has fewer total groups due to presence of a lot of events with no annotated positive relation linked to them  |
| epochs | 5 | 5 | — |
| batch_size | 16 | 16 | MPS OOM at higher values; CPU confirmed stable at 16 |
| device | cpu | cpu | MPS OOM; CPU gives identical results at the cost of time |
| seeds | {42, 1, 0} | {42, 1, 0} | ≥3 seeds for variance reporting |
| TripletLoss margin | 0.5 | 0.5 | — |
| Negative sampling | Fixed once at construction | same | Reproducibility; no per-epoch resampling |

**k=3 for temporal** (derivation from stats): avg_out=8.69 positives/event,
avg_not_out=22.85 negatives/event. Ratio ≈ 2.63 → ceiling integer = 3.
Full derivation in `src/ere_rag/data/stats_README.md`.

**k=5 for causal** (derivation from stats): avg_out=0.53, avg_not_out=31.0.
Raw ratio ≈ 58 (driven by extreme sparsity). k=5 chosen as a practical
upper bound to acknowledge the imbalance without overloading training.
Full derivation in `src/ere_rag/data/stats_README.md`.

### Build index and RQ1 sweep

```bash
python src/ere_rag/retriever/build_index.py \
    --category temporal \
    --output_dir experiments/retriever/temporal/<run_name> \
    --processed_dir data/processed \
    --percentages 5 10 20 30 50 60

python src/ere_rag/retriever/build_index.py \
    --category causal \
    --output_dir experiments/retriever/causal/<run_name> \
    --processed_dir data/processed \
    --percentages 5 10 20 30 50 60
```

Results and analysis: `experiments/retriever/temporal/README.md` and
`experiments/retriever/causal/README.md`.

**Operating point chosen for LLM stage:**
- Temporal: **60%** ; chosen to achieve retrieval recall ≈ 0.69, keeping
  ~69% of true positive candidates available to the LLM.
- Causal: **50%** same reasoning

**Partition size for LLM stage:**
- Temporal: **25** ≈ avg_retrieved at 60% (measured : 25)
- Causal: **20** ≈ avg_retrieved at 50% (measured: 16.6) 
Both values picked to ensure one partition per query event, it's a choice I made for these initial experiments but might later change it.

---

## Stage 2 — LLM Experiments

### Common settings (fixed across all LLM runs)

| Parameter | Value |
|---|---|
| Split | valid |
| n_docs | **20** (compute budget) |
| Retriever model | seed=42 for both categories |
| query_scope | `all_heads` |
| llm_backend | api |
| llm_model | qwen3-next-80b-a3b-instruct |
| temperature | 0.0 (deterministic) |
| max_new_tokens | 1600 |

### RQ2 and RQ3 configurations (4 per relation type)

RQ2 compares A vs B vs C (precision and call count/time).
RQ3 compares B vs D (precision; B doubles as the rationale-OFF arm).

| Config | Label | % cutoff | Partition size | Prompt | Purpose |
|---|---|---|---|---|---|
| A | single-pair + filtered| 60% / 50% | **1** | `llmere_few_shot_no_rationale` | RQ2: single-pair baseline |
| B | batched+filtered | 60% / 50% | 25 / 20 | `llmere_few_shot_no_rationale` | RQ2: proposed (no rationale); RQ3: rationale-OFF arm |
| C | batched+full-doc | **100%** | 25 / 20 | `llmere_few_shot_no_rationale` | RQ2: full-document baseline |
| D | batched+filtered+rationale | 60% / 50% | 25 / 20 | `llmere_few_shot` | RQ3: rationale-ON arm (also the full proposed pipeline / headline "ours") |

Config B is reported in both RQ2 and RQ3 tables — the same run, both tables, not two separate runs.

### Temporal experiment commands

```bash
TEMPORAL_DIR=experiments/retriever/temporal/grouped_temporal_k3_seed42_ep5_bs16_mg10000_cpu

# Config A — single-pair
python src/ere_rag/llm/pipeline.py \
    --run_dir $TEMPORAL_DIR --split valid --n_docs 20 \
    --percentage 0.60 --partition_size 1 \
    --prompt_version llmere_few_shot_no_rationale \
    --relation_type temporal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct

# Config B — batched+filtered, no rationale
python src/ere_rag/llm/pipeline.py \
    --run_dir $TEMPORAL_DIR --split valid --n_docs 20 \
    --percentage 0.60 --partition_size 25 \
    --prompt_version llmere_few_shot_no_rationale \
    --relation_type temporal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct

# Config C — batched+full-doc
python src/ere_rag/llm/pipeline.py \
    --run_dir $TEMPORAL_DIR --split valid --n_docs 20 \
    --percentage 1.0 --partition_size 25 \
    --prompt_version llmere_few_shot_no_rationale \
    --relation_type temporal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct

# Config D — batched+filtered, WITH rationale
python src/ere_rag/llm/pipeline.py \
    --run_dir $TEMPORAL_DIR --split valid --n_docs 20 \
    --percentage 0.60 --partition_size 25 \
    --prompt_version llmere_few_shot \
    --relation_type temporal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct
```

### Causal experiment commands

```bash
CAUSAL_DIR=experiments/retriever/causal/grouped_causal_k5_seed42_ep5_bs16_mg5000_cpu

# Config A — single-pair
python src/ere_rag/llm/pipeline.py \
    --run_dir $CAUSAL_DIR --split valid --n_docs 20 \
    --percentage 0.50 --partition_size 1 \
    --prompt_version llmere_few_shot_no_rationale \
    --relation_type causal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct

# Config B — batched+filtered, no rationale
python src/ere_rag/llm/pipeline.py \
    --run_dir $CAUSAL_DIR --split valid --n_docs 20 \
    --percentage 0.50 --partition_size 20 \
    --prompt_version llmere_few_shot_no_rationale \
    --relation_type causal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct

# Config C — batched+full-doc
python src/ere_rag/llm/pipeline.py \
    --run_dir $CAUSAL_DIR --split valid --n_docs 20 \
    --percentage 1.0 --partition_size 20 \
    --prompt_version llmere_few_shot_no_rationale \
    --relation_type causal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct

# Config D — batched+filtered, WITH rationale
python src/ere_rag/llm/pipeline.py \
    --run_dir $CAUSAL_DIR --split valid --n_docs 20 \
    --percentage 0.50 --partition_size 20 \
    --prompt_version llmere_few_shot \
    --relation_type causal --query_scope all_heads \
    --llm_backend api --llm_model qwen3-next-80b-a3b-instruct
```

Results and analysis: `experiments/llm/temporal/README.md` and
`experiments/llm/causal/README.md`.

---
