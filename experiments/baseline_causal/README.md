# Experiment: Temporal Relation Extraction Baseline

## Purpose
Reproduce the pairwise RoBERTa baseline from Wang et al. (2022)
on a 15% subset of MAVEN-ERE to validate the evaluation pipeline.

## Method
Original MAVEN-ERE pairwise classification code.
RoBERTa encodes all event pairs; feedforward classifier predicts
causal relation type (CAUSE, PRECONDITION, NONE).

## Paper reference
- Causal precision = 35.0 (full dataset, GPU, Wang et al. 2022)
- Causal recall = 27.2 (full dataset, GPU, Wang et al. 2022)
- Causal F1 = 30.6 (full dataset, GPU, Wang et al. 2022)

## Hardware constraint
All runs on Apple M1 (no CUDA). MPS used where possible.
Differences from paper numbers expected due to subset size and hardware.

## Runs summary

| Run | Device | Epochs | Batch Size | Best Val F1 | Precision | Recall |
|-----|--------|--------|----|-------------|-------------|-------------|
| seed42_sr0.15_ep10_bs1_cpu | cpu | 10 | 1 | 0.2133 | 0.2291 | 0.1995 |
| seed42_sr0.15_ep20_bs4_cpu | cpu | 20 | 4 | 0.2149 | 0.2519 | 0.1874 |

## Conclusion
Pipeline validated. Best local F1 ≈ 0.215.
