# Data

## Download

Run from the project root:

```bash
python src/ere-rag/data/download.py
```

This downloads MAVEN-ERE into `data/raw/MAVEN_ERE/`.

## Preprocessing

Run from the project root:

```bash
python src/ere-rag/data/preprocess.py
```

This creates the following files in `data/processed/`:
- `subset_train.jsonl` — training documents (15% subset, seed=42)
- `subset_valid.jsonl` — validation documents
- `train_pairs.jsonl` — event pairs with relation labels for training
- `valid_pairs.jsonl` — event pairs with relation labels for validation
- `stats.json` — dataset statistics

## Notes

- Do not commit any files from `data/raw/` or `data/processed/` — they are gitignored
- The subset uses random seed 42 for reproducibility
- Only temporal and causal relations are kept