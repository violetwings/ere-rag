# Retriever — Causal Relations

Model: `all-MiniLM-L6-v2` fine-tuned with TripletLoss (cosine, margin=0.5)
Category: causal | k_negatives: 5 | max_groups: 5,000 | epochs: 5 | device: cpu

For k justification see `src/ere_rag/data/stats_README.md`.
For evaluation setting explanation see `docs/methodology.md` (Datasets section).
---

## Training results (3 seeds)

| Seed | Triplet accuracy (best) | Training time (min) |
|---|---|---|
| 42 | 0.740 | 107.6 |
| 1  | 0.713 | 136.1 |
| 0  | 0.707 | 330.2 |
| **mean ± std** | **0.720 ± 0.018** | **191.3 ± 125.1** |

Triplet accuracy (0.71–0.74) contrasts with the temporal retriever's lower accuracy (0.582 ± 0.047) and
is consistent with causal sparsity (avg_norm_degree=0.025 (see `src/ere_rag/data/stats_README.md`) — only ~2.5% of a
document's events are causally related to a given anchor). Seed stability is good (std=0.018).
Seed=0 training took ~3× longer than the other two seeds (330 vs ~107–136 min) under identical config; this is attributed to an unattended overnight run.

---

## RQ1 sweep — retrieval P/R/F1 at each percentage (valid split, all event pairs)

**Evaluation note**: these numbers are computed in the *all event pairs* setting.
Retrieval "recall" means: fraction of all annotated true positive causal pairs
the retriever surfaces as candidates. It is an upper bound on downstream LLM recall.

### Seed 42

| % | P | R | F1 | avg_k |
|---|---|---|---|---|
| 5%  | 0.3151 | 0.1864 | 0.2343 | 1.7 |
| 10% | 0.2499 | 0.2814 | 0.2647 | 3.3 |
| 20% | 0.1828 | 0.4107 | 0.2530 | 6.6 |
| 30% | 0.1531 | 0.5154 | 0.2361 | 10.0 |
| 50% | 0.1229 | 0.6908 | 0.2087 | 16.6 |

### Seed 1

| % | P | R | F1 | avg_k |
|---|---|---|---|---|
| 5%  | 0.3136 | 0.1856 | 0.2331 | 1.7 |
| 10% | 0.2462 | 0.2772 | 0.2608 | 3.3 |
| 20% | 0.1766 | 0.3966 | 0.2443 | 6.6 |
| 30% | 0.1469 | 0.4946 | 0.2265 | 10.0 |
| 50% | 0.1192 | 0.6700 | 0.2024 | 16.6 |

### Seed 0

| % | P | R | F1 | avg_k |
|---|---|---|---|---|
| 5%  | 0.3148 | 0.1863 | 0.2341 | 1.7 |
| 10% | 0.2452 | 0.2761 | 0.2597 | 3.3 |
| 20% | 0.1768 | 0.3971 | 0.2446 | 6.6 |
| 30% | 0.1470 | 0.4948 | 0.2266 | 10.0 |
| 50% | 0.1185 | 0.6659 | 0.2012 | 16.6 |

### Mean ± std across seeds

| % | P mean±std | R mean±std | F1 mean±std |
|---|---|---|---|
| 5%  | 0.314 ± 0.001 | 0.186 ± 0.001 | 0.234 ± 0.001 |
| 10% | 0.247 ± 0.003 | 0.278 ± 0.003 | 0.262 ± 0.003 |
| 20% | 0.179 ± 0.003 | 0.401 ± 0.007 | 0.247 ± 0.005 |
| 30% | 0.149 ± 0.002 | 0.502 ± 0.011 | 0.229 ± 0.005 |
| 50% | 0.120 ± 0.002 | 0.682 ± 0.013 | 0.204 ± 0.004 |

Variance across seeds is very small (F1 std ≤ 0.005) — the causal retriever
is stable, consistent with its less variable triplet
accuracy.

**Notable pattern**: F1 peaks at 10% (0.262) and declines monotonically
afterward. This reflects causal sparsity (avg_norm_degree=0.025 , see
`src/ere_rag/data/stats_README.md`): past 10%,
added candidates are overwhelmingly non-related events, and precision falls
faster than recall gains compensate. For **RQ1 analysis**, the F1-optimal
operating point is 10%. For **RQ2/RQ3**, a higher percentage is used
to maximise recall for the downstream LLM (see below).

---

## Comparison with RoBERTa baseline (valid split, all event pairs setting)

**Primary comparison** uses our own reproduced numbers — same split, same 15%
training data, same evaluation setting.

| System | Causal P | Causal R | Causal F1 |
|---|---|---|---|
| RoBERTa [1] — **reproduced** (15% train, seed=42, valid) | 0.252 | 0.187 | 0.215 |
| RoBERTa [1] — paper reported (full train, full valid/test) | 0.350 | 0.272 | 0.306 |
| **Retriever @50%** (mean across seeds) | 0.120 | **0.682** | 0.204 |
| Retriever @10% (mean, F1-optimal) | 0.247 | 0.278 | 0.262 |

**At 50%**: retrieval recall (0.682) substantially exceeds both our reproduced
RoBERTa (0.187) and the paper's (0.272). This confirms the retriever successfully
surfaces a large fraction of causal positive pairs. However, precision (0.120) is
well below RoBERTa's (0.252 reproduced / 0.350 paper) — the retrieved pool
contains many non-related pairs. This is the expected tradeoff: the retriever is
a coarse filter; the LLM stage is responsible for separating true positives from
false positives within the retrieved pool.

**At 10%** (F1-optimal for retrieval): precision (0.247) approaches our reproduced
RoBERTa's level while recall (0.278) slightly exceeds it. However, this operating
point was not chosen for the LLM stage — see Operating Point below.

Our reproduced RoBERTa numbers are lower than the paper's (F1=0.215 vs 0.306)
because we trained on only 15% of the training data. This is an acknowledged
limitation.
---

## Operating point chosen for LLM stage

**Percentage: 50%** (measured; recall ≈ 0.682).

Rationale: at 10% (F1-optimal for retrieval), recall=0.278 — over 70% of true
causal relations are permanently lost before the LLM processes anything. The LLM
stage (RQ2/RQ3) tests whether filtering and batching affect *precision*; for this
to be meaningful, the LLM needs to see enough true positive candidates to classify
correctly. At 50%, recall=0.682, giving the LLM ~68% of all true causal pairs
as candidates. The higher noise in the retrieved pool (precision=0.120) is exactly
what RQ2 tests: can the LLM filter this noise while retaining precision?

The operating point for RQ2/RQ3 is thus intentionally not the F1-optimal
retrieval point — it deliberately prioritises recall headroom over retrieval
efficiency.

**Partition size: 20** ≈ avg_retrieved at 50% (16.6, rounded up). Goal: most
queries land in a single LLM call(it's a choice we made but might change it later)

**Retriever seed for LLM experiments: 42.**