"""
main.py

Causal Relation Extraction baseline on MAVEN-ERE.
Uses the original MAVEN-ERE pairwise classification approach.

Usage:
    python main.py --sample_rate 0.15 --epochs 20 --batch_size 1 \
                   --eval_steps 436 --log_steps 218 --seed 42 \
                   --device cpu
"""

import math
import sys
import os
from pathlib import Path

import torch
import yaml
import random
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
import argparse
from torch.optim import AdamW, Adam
import torch.nn as nn
from sklearn.metrics import classification_report
import warnings
import json

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parents[4]))

from src.ere_rag.baselines.causal.data import get_dataloader, REL2ID, ID2REL
from ere_rag.baselines.model import Model
from src.ere_rag.baselines.causal.dump_result import dump_result


def get_device(device_name):
    """
    Safely initialize selected device.
    """

    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("MPS requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")

    elif device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")

    return torch.device("cpu")


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate(model, dataloader, desc=""):
    pred_list = []
    label_list = []

    model.eval()

    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc):

            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

            scores = model(data)

            labels = data["labels"]

            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)

            pred = torch.argmax(scores, dim=-1)
            if labels.size(0) > 0:
                pred_list.extend(
                    pred[labels >= 0].cpu().numpy().tolist()
                )
                label_list.extend(
                    labels[labels >= 0].cpu().numpy().tolist()
                )

    report_labels = list(range(len(ID2REL) - 1))
    report_names = [ID2REL[i] for i in report_labels]

    result = classification_report(
        label_list,
        pred_list,
        output_dict=True,
        target_names=report_names,
        labels=report_labels
    )

    print(f"{desc} result:", result)

    return result


def predict(model, dataloader):
    all_preds = []

    model.eval()

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):

            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

            scores = model(data)

            labels = data["labels"]

            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)

            pred = torch.argmax(scores, dim=-1)

            max_label_length = data["max_label_length"]

            n_doc = len(labels) // max_label_length

            for i in range(n_doc):

                selected = labels[
                    i * max_label_length:(i + 1) * max_label_length
                ] >= 0

                all_preds.append({
                    "doc_id": data["doc_id"][i],
                    "preds": pred[
                        i * max_label_length:(i + 1) * max_label_length
                    ][selected].cpu().numpy().tolist(),
                })

    return all_preds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_steps", default=50, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--log_steps", default=50, type=int)

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bert_lr", default=1e-5, type=float)

    parser.add_argument("--batch_size", default=1, type=int)

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ignore_nonetype", action="store_true")

    parser.add_argument(
        "--sample_rate",
        default=None,
        type=float,
        help="Randomly sample a fraction of training data"
    )

    parser.add_argument(
        "--data_dir",
        default="data/raw/MAVEN_ERE",
        type=str,
        help="Path to MAVEN-ERE data directory"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device to use"
    )

    args = parser.parse_args()

    # ── Device selection ────────────────────────────────────────────────
    device = get_device(args.device)

    print("Using device:", device)

    # ── Output directory ────────────────────────────────────────────────
    device_name = str(device).replace(":", "_")

    output_dir = Path(
        f"experiments/baseline_causal/"
        f"seed{args.seed}_"
        f"sr{args.sample_rate}_"
        f"ep{args.epochs}_"
        f"bs{args.batch_size}_"
        f"{device_name}"
    )

    output_dir.mkdir(exist_ok=True, parents=True)

    # Redirect stdout to log file
    log_file = open(output_dir / "log.txt", "w")
    sys.stdout = log_file

    set_seed(args.seed)

    label_num = len(ID2REL)

    report_labels = list(range(len(ID2REL) - 1))
    report_names = [ID2REL[i] for i in report_labels]

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Loading data...")

    if not args.eval_only:

        train_dataloader = get_dataloader(
            tokenizer,
            "train",
            data_dir=args.data_dir,
            max_length=256,
            shuffle=True,
            batch_size=args.batch_size,
            ignore_nonetype=args.ignore_nonetype,
            sample_rate=args.sample_rate
        )

        dev_dataloader = get_dataloader(
            tokenizer,
            "valid",
            data_dir=args.data_dir,
            max_length=256,
            shuffle=False,
            batch_size=args.batch_size,
            ignore_nonetype=args.ignore_nonetype
        )

    test_dataloader = get_dataloader(
        tokenizer,
        "test",
        data_dir=args.data_dir,
        max_length=256,
        shuffle=False,
        batch_size=args.batch_size,
        ignore_nonetype=args.ignore_nonetype
    )

    print("Loading model...")

    model = Model(len(tokenizer), out_dim=label_num)

    model = model.to(device)

    Loss = nn.CrossEntropyLoss(ignore_index=-100)

    if not args.eval_only:

        bert_optimizer = AdamW(
            [
                p for p in model.encoder.model.parameters()
                if p.requires_grad
            ],
            lr=args.bert_lr
        )

        optimizer = Adam(
            [
                p for p in model.scorer.parameters()
                if p.requires_grad
            ],
            lr=args.lr
        )

        scheduler = get_linear_schedule_with_warmup(
            bert_optimizer,
            num_warmup_steps=200,
            num_training_steps=len(train_dataloader) * args.epochs
        )

        print("*" * 20 + "Start training" + "*" * 20)

        glb_step = 0

        best_score = 0.0
        best_precision = 0.0
        best_recall = 0.0

        train_losses = []
        pred_list = []
        label_list = []

        for epoch in range(args.epochs):

            for data in tqdm(
                train_dataloader,
                desc=f"Training epoch {epoch + 1}/{args.epochs}"
            ):

                model.train()

                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].to(device)

                scores = model(data)

                labels = data["labels"]

                scores = scores.view(-1, scores.size(-1))
                labels = labels.view(-1)

                loss = Loss(scores, labels)

                pred = torch.argmax(scores, dim=-1)

                pred_list.extend(
                    pred[labels >= 0].cpu().numpy().tolist()
                )

                label_list.extend(
                    labels[labels >= 0].cpu().numpy().tolist()
                )

                train_losses.append(loss.item())

                loss.backward()

                optimizer.step()
                bert_optimizer.step()

                scheduler.step()

                optimizer.zero_grad()
                bert_optimizer.zero_grad()

                glb_step += 1

                if glb_step % args.log_steps == 0:
                    print("*"*20 + "Train Prediction Examples" + "*"*20)
                    print("true:")
                    print(label_list[:20])
                    print("pred:")
                    print(pred_list[:20])
                    print(
                        f"Train {glb_step} steps: "
                        f"loss={np.mean(train_losses):.6f}"
                    )

                    res = classification_report(
                        label_list,
                        pred_list,
                        output_dict=True,
                        target_names=report_names,
                        labels=report_labels
                    )

                    print("Train result:", res)

                    train_losses = []
                    pred_list = []
                    label_list = []

                if glb_step % args.eval_steps == 0:

                    res = evaluate(
                        model,
                        dev_dataloader,
                        desc="Validation"
                    )

                    current_score = res["micro avg"]["f1-score"]

                    if current_score > best_score:

                        print("Best result so far!")

                        best_score = current_score
                        best_precision = res["micro avg"]["precision"]
                        best_recall = res["micro avg"]["recall"]

                        state = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        }

                        torch.save(state, output_dir / "best")

    print("*" * 30 + "Predict" + "*" * 30)

    best_path = output_dir / "best"

    if best_path.exists():

        print(f"Loading best model from {best_path}")

        state = torch.load(best_path, map_location=device)

        model.load_state_dict(state["model"])

    all_preds = predict(model, test_dataloader)

    dump_result(
        str(Path(args.data_dir) / "test.jsonl"),
        all_preds,
        output_dir
    )

    # Save config for reproducibility
    config = {
        'seed': args.seed,
        'sample_rate': args.sample_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_steps': args.eval_steps,
        'log_steps': args.log_steps,
        'lr': args.lr,
        'bert_lr': args.bert_lr,
        'ignore_nonetype': args.ignore_nonetype,
        'device': str(device),
        'data_dir': args.data_dir,
        'model': 'roberta-base',
    }

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    results = {
        'run': output_dir.name,
        'device': str(device),
        'seed': args.seed,
        'sample_rate': args.sample_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_steps': args.eval_steps,
        'best_checkpoint': {
            'selected_by': 'micro_avg_f1',
            'precision': best_precision,
            'recall': best_recall,
            'f1': best_score
        },
        'paper_reference': {
            'causal_precision': 57.8,
            'causal_recall': 53.9,
            'causal_f1': 55.8,
            'note': 'Wang et al. 2022, full dataset, GPU'
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log_file.close()