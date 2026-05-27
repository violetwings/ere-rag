# Experiment: Temporal Relation Extraction Baseline

## Purpose
Reproduce the pairwise RoBERTa baseline from Wang et al. (2022)
on a 15% subset of MAVEN-ERE to validate the evaluation pipeline.

## Method
Original MAVEN-ERE pairwise classification code.
RoBERTa encodes all event pairs; feedforward classifier predicts
temporal relation type (BEFORE, OVERLAP, CONTAINS, SIMULTANEOUS,
ENDS-ON, BEGINS-ON, NONE).

## Paper reference
- Temporal precision = 57.8 (full dataset, GPU, Wang et al. 2022)
- Temporal recall = 53.9 (full dataset, GPU, Wang et al. 2022)
- Temporal F1 = 55.8 (full dataset, GPU, Wang et al. 2022)

## Hardware constraint
All runs on Apple M1 (no CUDA). MPS used where possible.
Differences from paper numbers expected due to subset size and hardware.

## Runs summary

| Run | Device | Epochs | Batch Size | Best Val F1 | Precision | Recall |
|-----|--------|--------|----|-------------|-------------|-------------|
| seed42_sr0.15_ep5_bs1_mps | mps | 5 | 1 | 0.4723 | 0.4944 | 0.4520 |
| seed42_sr0.15_ep10_bs1_mps | mps | 10 | 1 | 0.4807 | 0.4752 | 0.4864 |
| seed42_sr0.15_ep10_bs4_cpu | cpu | 10 | 4 | 0.4884 | 0.4470 | 0.5381 |
| seed42_sr0.15_ep20_bs1_mps | mps | 20 | 1 | 0.4843 | 0.4536 | 0.5196 |

## Conclusion
Pipeline validated. Best local F1 ≈ 0.488.