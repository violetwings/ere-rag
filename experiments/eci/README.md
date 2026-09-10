# Refined LLM Stage

Retriever: seed=42, 50% cutoff (avg_k≈17), `grouped_causal_k5_seed42_ep5_bs16_mg5000_cpu`
LLM: Qwen3-Next-80B-A3B-Instruct (instruct mode, temperature=0.0)
Evaluation: valid split, n_docs=20

For config definitions (A/B/C/D) see `docs/experiment-plan-refined.md`.

---

## Results summary

| Config | % | Rationale | P | R | F1 | LLM calls | Time/doc (s) | Avg call (s) |
|---|---|---|---|---|---|---|---|---|---|
| A — per-query | 100% | No | 0.0658 | 0.6737 | 0.12 | 559 | 18.33 | 0.66 |
| B — per-query + filtered  | 50% | No | 0.0834 | 0.5751 | 0.1457 | 559 | 17.85 | 0.64 |
| C — whole document | 100% | No | 0.2124 | 0.1854 | 0.198 | 20 | 5.22 | 5.22 |
| D — whole document + reasoning | 100%| Yes | 0.3015 | 0.1408 | 0.192 | 20 | 6.42 | 6.42 |

Config C (whole document) appears in both the RQ2 and RQ3 tables —
it is the same run serving as the shared reference condition.

n_predictions = 559 for configs A/B (same number of query events);
n_predictions = 20 for configs C/D (same number of documents).

---

## RQ2 — per-query vs per-document calls, and filtered vs unfiltered

**per-query vs per-document calls (A vs C):**

| Metric | Per-query (A) | Per-document (C) | Change |
|---|---|---|---|
| LLM calls | 559| 20 | **−28×** |
| Time/doc (s) | 18.33 | 5.22 | **−3.5×** |
| Precision | 0.0658 | 0.2124 | **+0.1466 (+223%)** |
| Recall | 0.6737 | 0.1854 | −0.4883 (−73%) |
| F1 | 0.12 | 0.198 | +0.078 |

Per-document calls achieve a **28 reduction in calls** and a **3.5× speedup**.
It also shows a meaningful **precision gain (+223%)**
alongside the expected recall reduction. Per-query mode achieves higher recall
(0.6737) but at extremely low precision (0.0658) — for nearly every correct causal
prediction, ~15 incorrect predictions are made. Per-document classification improves this ratio
substantially (P=0.2124). The precision gain suggests that when the LLM is required to determine all possible causal links among events in a documents(sees multiple pairs together), it applies stricter criteria for what constitutes
a causal relation, reducing spurious predictions.

**Filtered vs Unflitered (A vs B):**

| Metric | Unfiltered 50% (A) | Filtered 50% (B) | Change |
|---|---|---|---|
| LLM calls | 559 | 559 | No change |
| Time/doc (s) | 18.33 | 17.85 | **−1.03×** |
| Precision | 0.0658 | 0.0834 | **+0.0176 (+26.7%)** |
| Recall | 0.6737 | 0.5751 | −0.0986 |
| F1 | 0.12 | 0.1457 | **+0.0257 (+21%)** |

Filtering to 50% gives a **26.7% precision improvement** and **21% F1 improvement** The unfiltered setting
provides however higher recall (more candidates means more chances to find the
true positives), but the precision advantage of filtering is decisive for F1.

This is a clear result for the filtering hypothesis in RQ2: for causal relations,
where true positives are extremely rare, presenting all
events to the LLM results in a very noisy pool where the LLM cannot reliably
distinguish positives from negatives. Filtering to the top 50% of candidates by
retrieval similarity substantially reduces this noise.


**Combined RQ2 verdict**: per-document classification provides strong efficiency gains  over per-query ECI and a
precision improvement. Filtering provides a clear F1 improvement by reducing causal noise.

---

## RQ3 — Reasoning vs no reasoning (C vs D)

| Metric | No reasoning (C) | Reasoning (D) | Change |
|---|---|---|---|
| Avg call time (s) | 5.22 | 6.42 | **+1.23× per call** |
| Precision | 0.2124 | 0.3015 | **+0.0891 (+42%)** |
| Recall | 0.1854 | 0.1408 | **-0.0446 (-24%)** |
| F1 | 0.198 | 0.192 | **30.006 (-3%)** |



---

