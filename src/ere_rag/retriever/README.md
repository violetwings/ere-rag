# Retriever — `src/ere_rag/retriever/`

Dense retrieval stage of the pipeline: fine-tunes an S-BERT bi-encoder per
relation category(temporal or causal or both), builds a FAISS index over event embeddings, and
evaluates retrieval recall/precision/F1 across a percentage sweep (RQ1).

## Files

- **`prepare_data.py`** — builds grouped `(anchor, positive, [negatives])`
  training examples from raw MAVEN-ERE, per category (`temporal` / `causal`
  / `both`). Uses the temporal `Document` class (events + TIMEX) for
  `temporal`/`both`, and the causal `Document` class (event mentions only,
  chain→mention expansion handled internally) for `causal`. Negatives are
  evenly distributed across each anchor's positives so none are wasted.

- **`train_retriever.py`** — fine-tunes `all-MiniLM-L6-v2` with TripletLoss
  on the groups from `prepare_data.py`. Samples `max_groups` across all documents in `interim_dir` and then `k_negatives` per group. Saves the best
  checkpoint (by `TripletEvaluator` accuracy on valid) plus `config.yaml`
  and `results.json` per run.

- **`build_index.py`** — encodes all candidate mentions per document with
  the fine-tuned model, builds a FAISS index, and evaluates recall@P% across
  a percentage sweep. Also writes a per-document candidate cache (`cache/`)
  consumed by the LLM stage, and saves embeddings/indices (`indices/`) for
  reuse without re-encoding.

## Running

Data-prep seed is always fixed at 42 (only the training seed varies — see
`docs/methodology.md`).

```bash
# 1. Prepare grouped training data (once per category)
python src/ere_rag/retriever/prepare_data.py \
    --category temporal \
    --data_dir data/raw/MAVEN_ERE \
    --output_dir data/interim/temporal \
    --sample_rate 0.15  \
    --seed 42

# 2. Train (repeated per seed: 42, 1, 0)
python src/ere_rag/retriever/train_retriever.py \
    --category temporal \
    --interim_dir data/interim/temporal \
    --experiments_dir experiments/retriever \
    --k_negatives 3 \
    --max_groups 10000 \
    --epochs 5 \
    --batch_size 16 \
    --device cpu \
    --seed 42


# 3. Build index + evaluate RQ1 sweep, (run_name_from_step_2 can be found under the experiments/retriever folder)
python src/ere_rag/retriever/build_index.py \
    --category temporal \
    --output_dir experiments/retriever/temporal/<run_name_from_step_2> \
    --processed_dir data/processed \
    --percentages 10 20 30 50 60
```

Same pattern for `--category causal` (used `--k_negatives 5 --max_groups
5000`, to increase positive-rate imbalance — see
`docs/methodology.md` for the justification).

`--category both` runs but has not been trained/evaluated this milestone —
following LLMERE's approach of treating temporal and causal separately.

## Output structure

```
experiments/retriever/<category>/<run_name>/
├── model/              # best checkpoint (by TripletEvaluator accuracy)
├── config.yaml          # full run config
├── results.json         # training summary (best triplet_accuracy, time)
├── rq1_results.json     # percentage-sweep P/R/F1 (from build_index.py)
├── indices/              # per-document FAISS index + embeddings + mentions
└── cache/                # per-document top-100 candidate cache (consumed by src/ere_rag/llm/)
```

## Known limitations

- Trained/evaluated on CPU only (MPS produced OOM errors on 8GB M1).
- `--category causal`'s candidate universe excludes TIMEX (causal relations
  never reference time expressions); `--category temporal`/`both` include it so the full tests on both categories have not been polished yet.
- Recall ceiling: any relation not retrieved at the chosen percentage is
  permanently unavailable to the downstream LLM stage.