"""
train_retriever.py

Fine-tunes S-BERT on MAVEN-ERE event relation groups using TripletLoss.

Reads the grouped format produced by prepare_data.py:
    { "anchor": str, "positive": str, "negatives": [str, ...], ... }

At each epoch, ONE negative is sampled at random from each group's
negatives list. This means:
  - The same (anchor, positive) pair sees a different negative each epoch
  - Over 5 epochs, the model explores up to 5 distinct negatives per pair
  - No negative is wasted or over-committed at data-prep time

Sampling (--max_groups):
  Groups are sampled proportionally per document so no single document
  dominates. The negative diversity is already baked in by prepare_data.py's
  even distribution — sampling groups preserves that structure.

Usage:
    python src/ere_rag/retriever/train_retriever.py \
        --max_groups 20000 \
        --epochs 1 \
        --batch_size 16 \
        --device mps
"""

import json
import random
import argparse
import time
import yaml
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
from sentence_transformers import (
    SentenceTransformer,
    losses,
    InputExample,
    evaluation,
)
from torch.utils.data import DataLoader, Dataset


# ── Device ─────────────────────────────────────────────────────────────────

def get_device(device_name: str) -> str:
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("MPS not available, falling back to CPU.")
        return "cpu"
    if device_name == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("CUDA not available, falling back to CPU.")
        return "cpu"
    return "cpu"


MARKER_TOKENS = ["<m>", "</m>"]


# ── Data loading ───────────────────────────────────────────────────────────

def load_groups(filepath: Path) -> list[dict]:
    groups = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    return groups


# ── Sampling — proportional per document ──────────────────────────────────

def sample_groups(
    groups:     list[dict],
    max_groups: int | None,
    seed:       int,
) -> list[dict]:
    """
    Sample up to max_groups, proportional per document so no single
    document dominates. Documents with many event pairs still contribute
    proportionally more — that's intentional, they have more signal.
    """
    if max_groups is None or len(groups) <= max_groups:
        return groups

    rng = random.Random(seed)

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        by_doc[g["doc_id"]].append(g)

    total   = len(groups)
    sampled = []

    for doc_groups in by_doc.values():
        budget = max(1, round(max_groups * len(doc_groups) / total))
        n      = min(len(doc_groups), budget)
        sampled.extend(rng.sample(doc_groups, n))

    # Trim or top-up to exact target
    if len(sampled) > max_groups:
        sampled = rng.sample(sampled, max_groups)
    elif len(sampled) < max_groups:
        used      = set(map(id, sampled))
        remaining = [g for g in groups if id(g) not in used]
        extra     = min(max_groups - len(sampled), len(remaining))
        if extra > 0:
            sampled.extend(rng.sample(remaining, extra))

    rng.shuffle(sampled)
    return sampled


# ── Per-epoch Dataset ──────────────────────────────────────────────────────

class GroupDataset(Dataset):
    """
    Each call to __getitem__ samples ONE negative at random from the
    group's negatives list. This means every epoch sees a different
    set of (anchor, positive, negative) triplets for the same groups,
    giving the model broader negative exposure without any extra storage.
    """

    def __init__(self, groups: list[dict], seed: int):
        self.groups = groups
        self.rng    = random.Random(seed)

    def reseed(self, epoch: int) -> None:
        """Call before each epoch to get different negatives per epoch."""
        self.rng = random.Random(epoch * 9999 + 42)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> InputExample:
        g       = self.groups[idx]
        neg     = self.rng.choice(g["negatives"])
        return InputExample(texts=[g["anchor"], g["positive"], neg])


# ── Evaluator ──────────────────────────────────────────────────────────────

def build_evaluator(
    valid_groups: list[dict],
    seed:         int,
    name:         str = "validation",
    max_eval:     int = 300,
) -> evaluation.TripletEvaluator:
    """
    For evaluation, fix one negative per group (first in list) so the
    metric is comparable across epochs. We use the first neg rather than
    random so the evaluator is deterministic.

    Expected accuracy range with hard in-document negatives: 0.70–0.85.
    Above 0.95 → negatives too easy. Below 0.60 → marker/label issue.
    """
    rng    = random.Random(seed)
    subset = valid_groups[:max_eval]
    return evaluation.TripletEvaluator(
        anchors=[g["anchor"]              for g in subset],
        positives=[g["positive"]          for g in subset],
        negatives=[g["negatives"][0]      for g in subset],  # fixed for comparability
        name=name,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune S-BERT with grouped negatives + TripletLoss"
    )
    parser.add_argument("--interim_dir",     type=str, default="data/interim")
    parser.add_argument("--experiments_dir", type=str, default="experiments/retriever")
    parser.add_argument("--base_model",      type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--epochs",          type=int, default=5)
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="16 works well on MPS. Reduce to 8 if OOM."
    )
    parser.add_argument("--warmup_steps",    type=int, default=100)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["cpu", "mps", "cuda"]
    )
    parser.add_argument(
        "--max_groups", type=int, default=20000,
        help="Max groups to train on per run. Each epoch re-samples negatives."
    )
    args = parser.parse_args()

    # ── Output dir ─────────────────────────────────────────────────────
    run_name = (
        f"grouped"
        f"_seed{args.seed}"
        f"_ep{args.epochs}"
        f"_bs{args.batch_size}"
        f"_mg{args.max_groups}"
        f"_{args.device}"
    )
    output_dir = Path(args.experiments_dir) / run_name
    model_dir  = output_dir / "model"
    eval_dir = model_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run:    {run_name}")
    print(f"Output: {output_dir}")

    # ── Load groups ────────────────────────────────────────────────────
    print("\nLoading groups...")
    train_groups = load_groups(Path(args.interim_dir) / "train_groups.jsonl")
    valid_groups = load_groups(Path(args.interim_dir) / "valid_groups.jsonl")
    print(f"  Train: {len(train_groups):,}  |  Valid: {len(valid_groups):,}")

    avg_neg = sum(g["n_neg"] for g in train_groups) / max(1, len(train_groups))
    print(f"  Avg negatives per group (train): {avg_neg:.1f}")

    # ── Sample groups ──────────────────────────────────────────────────
    print(f"\nSampling to max_groups={args.max_groups}...")
    train_groups = sample_groups(train_groups, args.max_groups, args.seed)
    print(f"  Using: {len(train_groups):,} groups")
    n_docs = len({g["doc_id"] for g in train_groups})
    print(f"  Docs represented: {n_docs}  "
          f"| avg groups/doc: {len(train_groups)/max(1,n_docs):.1f}")

    # ── Dataset + model ────────────────────────────────────────────────
    dataset = GroupDataset(train_groups, seed=args.seed)

    device = get_device(args.device)
    print(f"\nLoading '{args.base_model}' on {device}...")
    model  = SentenceTransformer(args.base_model, device=device)

    wem = model._first_module()
    wem.tokenizer.add_tokens(MARKER_TOKENS, special_tokens=True)
    wem.auto_model.resize_token_embeddings(len(wem.tokenizer))
    print(f"  Registered marker tokens: {MARKER_TOKENS}")
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Loss ───────────────────────────────────────────────────────────
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.5,
    )

    # ── Evaluator ──────────────────────────────────────────────────────
    evaluator = build_evaluator(valid_groups, seed=args.seed, max_eval=300)

    # ── Manual training loop (needed for per-epoch negative resampling) ─
    # We can't use model.fit() here because we need to call dataset.reseed()
    # before each epoch's DataLoader is created. Instead we replicate
    # model.fit()'s core logic manually.

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    steps_per_epoch = len(train_groups) // args.batch_size
    total_steps     = steps_per_epoch * args.epochs
    est_min         = (total_steps * 1.5) / 60

    print(f"\nTraining with per-epoch negative resampling...")
    print(f"  Loss:             TripletLoss (cosine, margin=0.5)")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Groups/epoch:     {len(train_groups):,}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Steps/epoch:      {steps_per_epoch}  |  Total: {total_steps}")
    print(f"  Est. time:        ~{est_min:.0f} min on M1")
    print(f"  Negative variety: up to {args.epochs} distinct negatives "
          f"per (anchor, positive) pair across epochs")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    best_score  = -1.0
    start       = time.time()

    for epoch in range(args.epochs):
        # Reseed dataset so this epoch draws different negatives
        dataset.reseed(epoch)

        dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=lambda batch: batch,   # list of InputExamples
        )

        model.train()
        epoch_loss = 0.0
        n_steps    = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
            # Encode the three texts in each InputExample
            texts_a = [ex.texts[0] for ex in batch]
            texts_p = [ex.texts[1] for ex in batch]
            texts_n = [ex.texts[2] for ex in batch]

            # Get sentence embeddings through the model
            features_a = model.tokenize(texts_a)
            features_p = model.tokenize(texts_p)
            features_n = model.tokenize(texts_n)

            for feat in [features_a, features_p, features_n]:
                for k in feat:
                    if isinstance(feat[k], torch.Tensor):
                        feat[k] = feat[k].to(device)

            
            loss = train_loss(
                [features_a, features_p, features_n],
                labels=None,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_steps    += 1

        avg_loss = epoch_loss / max(1, n_steps)

        # Evaluate
        score = evaluator(model, output_path=str(eval_dir), epoch=epoch, steps=-1)
        print(f"  Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
              f"triplet_acc={score:.4f}")

        if score > best_score:
            best_score = score
            model.save(str(model_dir))
            print(f"    ✓ New best ({best_score:.4f}) — model saved")

    train_time = time.time() - start
    print(f"\nDone in {train_time/60:.1f} min. Best model at {model_dir}")

    # ── Save config + results ──────────────────────────────────────────
    config = {
        "loss":                  "TripletLoss",
        "triplet_margin":        0.5,
        "distance_metric":       "cosine",
        "negative_strategy":     "grouped per-anchor, one drawn per epoch",
        "base_model":            args.base_model,
        "epochs":                args.epochs,
        "batch_size":            args.batch_size,
        "warmup_steps":          args.warmup_steps,
        "max_groups":            args.max_groups,
        "seed":                  args.seed,
        "device":                device,
        "marker_tokens":         MARKER_TOKENS,
        "train_groups_used":     len(train_groups),
        "training_time_minutes": round(train_time / 60, 1),
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    results = {
        "run":                   run_name,
        "best_triplet_accuracy": round(best_score, 4),
        "training_time_minutes": round(train_time / 60, 1),
        "recall_at_k":           "to be filled by build_index.py",
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Config and results → {output_dir}")


if __name__ == "__main__":
    main()
