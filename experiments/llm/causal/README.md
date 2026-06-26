# LLM Stage — Causal Relations

Retriever: seed=42, 50% cutoff (avg_k≈17), `grouped_causal_k5_seed42_ep5_bs16_mg5000_cpu`
LLM: Qwen3-Next-80B-A3B-Instruct (instruct mode, temperature=0.0)
Evaluation: valid split, n_docs=20, query_scope=all_heads, all event pairs setting

For config definitions (A/B/C/D) see `docs/experiment-plan-m1.md`.

---

## Results summary

| Config | % | PS | Rationale | P | R | F1 | LLM calls | Time/doc (s) | Avg call (s) |
|---|---|---|---|---|---|---|---|---|---|
| A — single-pair+filtered | 50% | 1 | No | 0.0162 | 0.2653 | 0.0305 | 10,805 | 259 | 0.48 |
| B — batched+filtered | 50% | 20 | No | 0.0220 | 0.1690 | 0.0389 | 843 | 26 | 0.61 |
| C — batched+full-doc | 100% | 20 | No | 0.0140 | 0.1948 | 0.0261 | 1,320 | 54 | 0.81 |
| D — batched+filtered+rationale | 50% | 20 | Yes | 0.0247 | 0.1972 | 0.0439 | 843 | 127 | 3.02 |

Config B (batched+filtered, no rationale) appears in both the RQ2 and RQ3 tables —
it is the same run serving as the shared reference condition.

n_predictions = 10,805 for configs A/B/D (50% cutoff, same candidate pool);
n_predictions = 21,554 for config C (100% cutoff, all event pairs as candidates).

---

## RQ2 — Batch vs single-pair, and filtered vs full-document

**Batch vs single-pair (A vs B):**

| Metric | Single (A) | Batched (B) | Change |
|---|---|---|---|
| LLM calls | 10,805 | 843 | **−12.8×** |
| Time/doc (s) | 259 | 26 | **−10.0×** |
| Precision | 0.0162 | 0.0220 | **+0.0058 (+36%)** |
| Recall | 0.2653 | 0.1690 | −0.0963 (−36%) |
| F1 | 0.0305 | 0.0389 | +0.0084 |

Batching achieves a **12.8× reduction in calls** and a **10× speedup**.
Unlike temporal, causal batching shows a meaningful **precision gain (+36%)**
alongside the expected recall reduction. Single-pair mode achieves higher recall
(0.265) but at extremely low precision (0.016) — for nearly every correct causal
prediction, ~60 incorrect predictions are made. Batching improves this ratio
substantially (P=0.022). The precision gain suggests that when the LLM sees
multiple candidates together, it applies stricter criteria for what constitutes
a causal relation, reducing spurious predictions.

**Filtered vs full-document (B vs C):**

| Metric | Filtered 50% (B) | Full doc 100% (C) | Change |
|---|---|---|---|
| LLM calls | 843 | 1,320 | −36% |
| Time/doc (s) | 26 | 54 | **−2.1×** |
| Precision | 0.0220 | 0.0140 | **+0.0080 (+57%)** |
| Recall | 0.1690 | 0.1948 | −0.0258 |
| F1 | 0.0389 | 0.0261 | **+0.0128 (+49%)** |

Filtering to 50% gives a **57% precision improvement** and **49% F1 improvement**
over full-document partitioning, with 2.1× fewer calls. The full-document setting
provides however higher recall (more candidates means more chances to find the
true positives), but the precision advantage of filtering is decisive for F1.

This is a clear result for the filtering hypothesis in RQ2: for causal relations,
where true positives are extremely rare (~2.5% of event pairs), presenting all
events to the LLM results in a very noisy pool where the LLM cannot reliably
distinguish positives from negatives. Filtering to the top 50% of candidates by
retrieval similarity substantially reduces this noise.


**Combined RQ2 verdict**: batching provides strong efficiency gains and a
precision improvement for causal. Filtering provides a clear F1 improvement
(49%) by reducing causal noise. The combination (Config B) substantially
outperforms both full-document and single-pair baselines in F1.

---

## RQ3 — Rationale vs no rationale (B vs D)

| Metric | No rationale (B) | Rationale (D) | Change |
|---|---|---|---|
| Avg call time (s) | 0.61 | 3.02 | **+4.9× per call** |
| Time/doc (s) | 26 | 127 | **+4.9× total** |
| Precision | 0.0220 | 0.0247 | **+0.0027 (+12%)** |
| Recall | 0.1690 | 0.1972 | **+0.0282 (+17%)** |
| F1 | 0.0389 | 0.0439 | **+0.0050 (+13%)** |



---

## Comparison with baselines

| System | Causal P | Causal R | Causal F1 |
|---|---|---|---|
| RoBERTa [1] reproduced (15% train, valid) | 0.252 | 0.187 | 0.215 |
| RoBERTa [1] paper | 0.350 | 0.272 | 0.306 |
| LLMERE [3] paper (Llama3-8b-instruct) | 34.9 | 36.1 | 35.5 |
| **Ours — Config D (full pipeline)** | 0.0247 | 0.1972 | 0.0439 |
| **Ours — Config B (batched+filtered)** | 0.0220 | 0.1690 | 0.0389 |
| **Ours — Config C (full-doc baseline)** | 0.0140 | 0.1948 | 0.0261 |

