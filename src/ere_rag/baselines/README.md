# Baselines

This folder contains baseline implementations for reproducing experiments on the MAVEN-ERE dataset.

Two tasks are currently implemented:

- `temporal/` → Temporal relation extraction
- `causal/` → Causal relation extraction

Each task contains its own:
- `main.py`
- `data.py`
- `dump_result.py`
- task-specific label mappings and preprocessing

---

# Project Structure

```text
baselines/
│
├── temporal/
│   ├── main.py
│   ├── data.py
│   └── dump_result.py
│
├── causal/
│   ├── main.py
│   ├── data.py
│   └── dump_result.py
│
├── utils.py
└── model.py
```

---

# Running Experiments

Experiments must be launched using the `main.py` corresponding to the desired task.

---

# Temporal Relation Extraction

Run from the repository root:

```bash
python src/ere_rag/baselines/temporal/main.py \
    --sample_rate 0.15 \
    --epochs 20 \
    --batch_size 1 \
    --eval_steps 436 \
    --log_steps 218 \
    --seed 42 \
    --device cpu
```

Example using Apple Silicon GPU (MPS):

```bash
python src/ere_rag/baselines/temporal/main.py \
    --sample_rate 0.15 \
    --epochs 20 \
    --batch_size 1 \
    --eval_steps 436 \
    --log_steps 218 \
    --seed 42 \
    --device mps
```

---

# Causal Relation Extraction

Run from the repository root:

```bash
python src/ere_rag/baselines/causal/main.py \
    --sample_rate 0.15 \
    --epochs 20 \
    --batch_size 4 \
    --eval_steps 100 \
    --log_steps 20 \
    --seed 42 \
    --device cpu
```

---

# Main Arguments

| Argument | Description |
|---|---|
| `--sample_rate` | Fraction of training data used |
| `--epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--eval_steps` | Validation frequency (in training steps) |
| `--log_steps` | Logging frequency |
| `--seed` | Random seed |
| `--device` | `cpu`, `mps`, or `cuda` |
| `--eval_only` | Run evaluation only |
| `--ignore_nonetype` | Ignore NONE relation class |
| `--data_dir` | Path to MAVEN-ERE dataset |

---

# Devices

Supported devices:

- `cpu`
- `mps` (Apple Silicon GPU acceleration)
- `cuda` (NVIDIA GPU acceleration)

Example:

```bash
--device cpu
```

or

```bash
--device mps
```

---

# Outputs

Each run creates an experiment folder containing:

- `log.txt` → training and evaluation logs
- `best` → best model checkpoint
- `config.yaml` → experiment configuration
- `results.json` → summarized metrics
- `test_predictions.jsonl` → prediction outputs

Example:

```text
experiments/baseline_temporal/
└── seed42_sr0.15_ep20_bs1_cpu/
```

---

# Notes

- Best checkpoints are selected using validation micro-F1.
- Results may vary depending on:
  - sample rate
  - random seed
  - hardware
  - device backend (`cpu`, `mps`, `cuda`)
- Reported MAVEN-ERE paper scores are obtained on the full dataset using GPU training.