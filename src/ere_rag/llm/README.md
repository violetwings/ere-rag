# LLM Stage — `src/ere_rag/llm/`

Consumes the retriever's cached candidates and runs partitioned batch LLM
classification with optional rationale generation (RQ2/RQ3).

## Files

- **`pipeline.py`** — orchestrates everything: loads cached retrieval
  per document, selects candidates (by `--k` or `--percentage`), partitions
  them, builds prompts, calls the LLM, parses output, evaluates against
  ground truth, and saves predictions/timing/results.

- **`prompt.py`** — builds prompts (requires llm to generate relation-subtype
  buckets, e.g. `BEFORE: <e2>, <e5>`, rather than one required line per
  candidate). `TEMPLATE_CONFIG` defines each `--prompt_version` as a
  combination of four flags: `strict` (tightened definitions), `rationales`
  (on/off), `examples` (few-shot on/off): the few shot part is only to indicate how the format of the output shoud be, not to help give better predictions(so it needs to be refined), `descriptions` (full relation
  definitions vs. label names only).

- **`parse_output.py`** — parses LLM output into one prediction per
  (candidate, relation_category) pair. Handles both the current bucket
  format and the older `e0_temporal: LABEL | reason` line format for
  backward compatibility with earlier experiment outputs.

- **`inference.py`** — backend dispatch: `groq`, `api` (generic
  OpenAI-compatible — what we use for Qwen3-Next-80B-A3B-Instruct),
  `ollama`, `hf`. Includes retry/backoff for rate limits.


## Backend setup

For `--llm_backend api` (Qwen3-Next-80B-A3B-Instruct via hosted API):
```bash
export ERE_LLM_API_KEY="..."
export ERE_LLM_API_BASE="https://your-endpoint/v1"
```

## Running

Final RQ2/RQ3 configuration (see `experiment-plan-m1.md` for the full
4-config-per-relation-type design and reasoning):

```bash
# <run_name>: name of the run folder created after the retriever step , e.g: grouped_temporal_k3_seed42_ep5_bs16_mg10000_cpu 
#<category>: temporal or causal
#prompt_version: currently only using "llmere_few_shot" and "llmere_few_shot_no_rationale"
#percentage : percentage of candidate events retrieved per query event per document
python src/ere_rag/llm/pipeline.py \
    --run_dir experiments/retriever/<category>/<run_name>  \
    --split valid \
    --n_docs <N> \
    --percentage 0.50 \
    --partition_size 25 \
    --prompt_version llmere_few_shot \
    --relation_type <category> \
    --query_scope all_heads \
    --llm_backend api \
    --llm_model qwen3-next-80b-a3b-instruct \
    --sample_seed 42
```

Swap `--prompt_version llmere_few_shot_no_rationale` for configs A/B/C,
`--percentage 1.0` for config C (100% — no filtering), `--partition_size 1`
for config A (single-pair). Mirror with `--relation_type causal`,
`--partition_size 20`, and the causal `--run_dir`.

## Output structure

```
experiments/llm/<relation_type>/<retriever_run_name>/
└──<llm_backend>_<model_tag>_nd<N>_pct<P>_split<split>_pv<prompt_version>_ps<partition_size>_<query_scope>/
    ├── predictions.jsonl   # one row per (head, tail, category) prediction
    ├── timing.jsonl        # per-call timing
    └── results.json        # aggregate P/R/F1 + timing summary
```
