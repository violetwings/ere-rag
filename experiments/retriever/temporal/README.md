# Retriever — Temporal Relations

Model: `all-MiniLM-L6-v2` fine-tuned with TripletLoss (cosine, margin=0.5)
Category: temporal | k_negatives: 3 | max_groups: 10,000 | epochs: 5 | device: cpu

For k justification see `src/ere_rag/data/stats_README.md`.
For evaluation setting explanation see `docs/methodology.md` (Datasets section).

---

## Training results (3 seeds)

| Seed | Triplet accuracy (best) | Training time (min) |
|---|---|---|
| 42 | 0.560 | 129.4 |
| 1  | 0.637 | 203.4 |
| 0  | 0.550 | 95.5 |
| **mean ± std** | **0.582 ± 0.047** | **142.8 ± 55.7** |

The retriever is
learning some signal but struggling with the difficulty of in-document
negatives for temporal relations as suggested by the average accuracy value of 0.582.
Variance across seeds ,std=0.047  is moderate implying that training is not entirely
unstable, but seed matters.
 
---

## RQ1 sweep — retrieval P/R/F1 at each percentage (valid split, all event pairs)

**Evaluation note**: these numbers are computed in the *all event pairs* setting,
meaning all candidate pairs in each document (including true-NOREL pairs) form
the evaluation universe. Retrieval "recall" here means: fraction of all annotated
true positive temporal pairs that the retriever surfaces as candidates. It is an
upper bound on what the downstream LLM classifier can achieve.

### Seed 42

| % | P | R | F1 | avg_k |
|---|---|---|---|---|
| 5% | 0.4549 | 0.0721 | 0.1245 | 2.1 |
| 10% | 0.4296 | 0.1359 | 0.2065 | 4.2 |
| 20% | 0.4121 | 0.2606 | 0.3193 | 8.3 |
| 30% | 0.3993 | 0.3781 | 0.3884 | 12.5 |
| 50% | 0.3773 | 0.5962 | 0.4621 | 20.8 |
| 60% | 0.367 | 0.6967 | 0.4808 | 25.0 |

### Seed 1

| % | P | R | F1 | avg_k |
|---|---|---|---|---|
| 5% | 0.4659 | 0.0739 | 0.1275 | 2.1 |
| 10% | 0.4408 | 0.1395 | 0.2119 | 4.2 |
| 20% | 0.4192 | 0.2650 | 0.3248 | 8.3 |
| 30% | 0.4066 | 0.3850 | 0.3955 | 12.5 |
| 50% | 0.3860 | 0.6100 | 0.4728 | 20.8 |
| 60% | 0.3744 | 0.7107 | 0.4905 | 25.0 |

### Seed 0

| % | P | R | F1 | avg_k |
|---|---|---|---|---|
| 5% | 0.4574 | 0.0725 | 0.1252 | 2.1 |
| 10% | 0.4336 | 0.1372 | 0.2084 | 4.2 |
| 20% | 0.4137 | 0.2616 | 0.3205 | 8.3 |
| 30% | 0.4012 | 0.3799 | 0.3903 | 12.5 |
| 50% | 0.3776 | 0.5967 | 0.4625 | 20.8 |
| 60% | 0.3663 | 0.6952 | 0.4798 | 25.0 |

### Mean ± std across seeds

| % | P mean±std | R mean±std | F1 mean±std |
|---|---|---|---|
| 5% | 0.459 ± 0.006 | 0.073 ± 0.001 | 0.126 ± 0.002 |
| 10% | 0.435 ± 0.006 | 0.137 ± 0.002 | 0.208 ± 0.003 |
| 20% | 0.415 ± 0.004 | 0.262 ± 0.002 | 0.321 ± 0.003 |
| 30% | 0.402 ± 0.004 | 0.381 ± 0.004 | 0.391 ± 0.004 |
| 50% | 0.380 ± 0.005 | 0.601 ± 0.007 | 0.465 ± 0.006 |
| 60% | 0.369 ± 0.005 | 0.701 ± 0.009 | 0.484 ± 0.006 |

Seed variance is small across all percentages (F1 std ≤ 0.006), confirming that
retrieval performance is stable across training seeds despite triplet accuracy
varying more. 
Recall increases monotonically with percentage while precision
decreases monotonically.

---

## Comparison with RoBERTa baseline (valid split, all event pairs setting)

**Primary comparison** uses our own reproduced numbers — same split, same 15%
training data, same evaluation setting. The paper's numbers are included as
a reference point, noting they trained on the full dataset.

| System | Temporal P | Temporal R | Temporal F1 |
|---|---|---|---|
| RoBERTa [1] — **reproduced** (15% train, seed=42, valid) | 0.4470 | 0.5381 | 0.4884 |
| RoBERTa [1] — paper reported (full train, full valid/test) | 0.578 | 0.539 | 0.558 |
| **Retriever @60%** (mean across seeds) | 0.369 | **0.701** | 0.484 |
| Retriever @30% (mean across seeds) | 0.402 | 0.381 | 0.391 |

These quantities measure different things: RoBERTa's P/R/F1 come from a classifier
making label predictions; the retriever's P/R/F1 measure whether the retriever
*surfaced* the right pairs as candidates. They are not directly comparable, but
they share the same denominator for recall (all true positive temporal pairs),
so the recall comparison is meaningful.

**Key finding**: at 60%, retrieval recall (0.701) **substantially exceeds**
our reproduced RoBERTa's recall (0.538) and even the paper's reported recall
(0.539). This confirms the retriever successfully ranks true positive pairs
highly — nearly 70% of all temporal relations in the document appear in the
top 60% of candidates. Retrieval precision (0.369) is lower than RoBERTa's,
which is expected: the retriever's role is to recall candidates broadly, not
to classify them precisely. The LLM stage is responsible for precision.

Our reproduced RoBERTa numbers are lower than the paper's (F1=0.488 vs 0.558)
because we trained on only 15% of the training data. This is an acknowledged
limitation — see `docs/methodology.md` (Threats to Validity).

---

## Operating point chosen for LLM stage

**Percentage: 60%**

Rationale: RQ2 and RQ3 test what the LLM does with the retrieved candidates.
For this to be a meaningful test, the LLM must see a large fraction of the
true positive pairs. At 30% (F1-optimal for retrieval), recall=0.381 — over
60% of true relations have already been permanently lost before the LLM
processes anything. At 60%, recall=0.701 — the LLM receives ~70% of all
true temporal pairs as candidates. This provides sufficient headroom to test
whether filtering and batching affect precision, which is what RQ2 measures.

**Partition size: 25** ≈ avg_retrieved at 60% (extrapolated from 20.8 at
50%). Goal: most queries land in a single LLM call(it's just a choice we made for now but could be changed later)

**Retriever seed used for LLM experiments: 42.** 
Seed 1 achieved slightly better results (0.711 vs 0.697 at 60%) but the difference across all the different seeds is pretty negligeable (std ≈ 0.007) so we just chose to sitck to seed 42 
for consistency with the data-prep seed and the RoBERTa baseline.
