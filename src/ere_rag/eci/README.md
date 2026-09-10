# Refined LLM Stage — `src/ere_rag/eci/`

Consumes the retriever's cached candidates for the filtered mode and runs per-query/per-document LLM ECI with optional reasoning generation (RQ2/RQ3).

## Files

- **`pipeline.py`** — orchestrates everything: samples and loads documents and in the case of the filtering mode loads cached retrieval per document and selects candidates (by `--k` or `--percentage`), marks the document events, builds prompts, calls the LLM, parses output, evaluates against ground truth, and saves predictions/timing/results.

- **`prompt.py`** — builds prompts based on the selected mode and picks the appropriate few-shot examples to be used.

- **`parse_output.py`** — parses LLM output into one prediction per
  (candidate, relation_category) pair. Handles both the current bucket
  format and the older `e0_temporal: LABEL | reason` line format for
  backward compatibility with earlier experiment outputs.

- **`inference.py`** — backend dispatch: `groq`, `api` (generic
  OpenAI-compatible — what we use for Qwen3-Next-80B-A3B-Instruct),
  `ollama`, `hf`. Includes retry/backoff for rate limits.

- **`document_text.py`** — build the marked document text.

## Backend setup

For `--llm_backend api` (Qwen3-Next-80B-A3B-Instruct via hosted API):
```bash
export ERE_LLM_API_KEY="..."
export ERE_LLM_API_BASE="https://your-endpoint/v1"
```

## Running

Final RQ2/RQ3 configuration (see `experiment-plan-refined.md` for the full
4-config-per-relation-type design and reasoning):

```bash
# <run_name>: name of the run folder created after the retriever step , e.g: grouped_temporal_k3_seed42_ep5_bs16_mg10000_cpu 
#mode: run mode among ["whole_doc_reasoning", "whole_doc_no_reasoning", "per_query_no_retriever", "per_query_retriever"]
#percentage : to be used only in per_query_retriever mode: the percentage of candidate events retrieved per query event per document and presented to the llm 
python src/ere_rag/eci/pipeline.py \
    --raw_data_path data/raw/MAVEN_ERE/valid.jsonl \
    --processed_dir data/processed \
    --run_dir experiments/retriever/<run_name> \
    --split valid \
    --n_docs 20 \
    --percentage <percentage> \
    --mode <mode> \
    --llm_backend api \
    --llm_model qwen3-next-80b-a3b-instruct
```

Use `--mode per_query_no_retriever` for config A
`--mode per_query_retriever` for config B with `--percentage 0.5`
`--mode whole_doc_no_reasoning` for config C 
and `--mode whole_doc_reasoning` for config D

## Output structure

```
experiments/eci/<mode>/
└──<llm_backend>_<model_tag>_nd<N>_pct<P>_split<split>
    ├── predictions.jsonl   # one row per (head, tail, category) prediction
    ├── timing.jsonl        # per-call timing
    └── results.json        # aggregate P/R/F1 + timing summary
```
