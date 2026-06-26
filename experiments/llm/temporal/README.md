# LLM Stage — Temporal Relations

Retriever: seed=42, 60% cutoff (avg_k≈25), `grouped_temporal_k3_seed42_ep5_bs16_mg10000_cpu`
LLM: Qwen3-Next-80B-A3B-Instruct (instruct mode, temperature=0.0)
Evaluation: valid split, n_docs=20, query_scope=all_heads, all event pairs setting

For config definitions (A/B/C/D) see `docs/experiment-plan-m1.md`.

---

## Results summary

| Config | % | PS | Rationale | P | R | F1 | LLM calls | Time/doc (s) | Avg call (s) |
|---|---|---|---|---|---|---|---|---|---|
| A — single-pair+filtered | 60% | 1 | No | 0.2036 | 0.4371 | 0.2778 | 20,282 | 649 | 0.64 |
| B — batched+filtered | 60% | 25 | No | 0.2002 | 0.3669 | 0.2590 | 1,144 | 86 | 1.50 |
| C — batched+full-doc | 100% | 25 | No | 0.1781 | 0.5460 | 0.2686 | 1,655 | 120 | 1.45 |
| D — batched+filtered+rationale | 60% | 25 | Yes | 0.1958 | 0.3332 | 0.2466 | 1,144 | 380 | 6.65 |

Config B (batched+filtered, no rationale) appears in both the RQ2 and RQ3 tables —
it is the same run serving as the shared reference condition.

---

## RQ2 — Batch vs single-pair, and filtered vs full-document

**Batch vs single-pair (A vs B):**

| Metric | Single (A) | Batched (B) | Change |
|---|---|---|---|
| LLM calls | 20,282 | 1,144 | **−17.7×** |
| Time/doc (s) | 649 | 86 | **−7.6×** |
| Precision | 0.2036 | 0.2002 | −0.0034 (−1.7%) |
| Recall | 0.4371 | 0.3669 | −0.0702 (−16.1%) |
| F1 | 0.2778 | 0.2590 | −0.0188 |

Batching achieves a **17.7× reduction in LLM calls** and a **7.6× speedup**
with negligible precision loss (−0.0034). The main cost is recall: presenting
candidates in groups of 25 causes the LLM to miss ~16% of the relations it
would have found in single-pair mode. This is a genuine precision/recall tradeoff
inherent to batching — in a large partition, some lower-prominence relations may be
lost in the context of more prominent ones.

**Filtered vs full-document (B vs C):**

| Metric | Filtered 60% (B) | Full doc 100% (C) | Change |
|---|---|---|---|
| LLM calls | 1,144 | 1,655 | −31% |
| Time/doc (s) | 86 | 120 | −28% |
| Precision | 0.2002 | 0.1781 | **+0.0221 (+12.4%)** |
| Recall | 0.3669 | 0.5460 | −0.0791 (−14.5%) |
| F1 | 0.2590 | 0.2686 | −0.0096 |

Filtering to 60% improves precision by **12.4%** compared to full-document
partitioning, with 31% fewer calls. Recall drops possibly partially because the retriever misses
~30% of true temporal relations at the 60% cutoff (retrieval recall=0.697).
F1 is marginally lower for the filtered condition, suggesting that for temporal
relations the full-document approach's recall advantage outweighs the precision
gain from filtering at this cutoff. The **precision improvement from filtering
is the cleaner finding** — consistent with the noise-reduction hypothesis in RQ2.

**Combined RQ2 verdict**: batching provides clear efficiency gains (calls and
wall time) with acceptable precision and worse recall trade-offs. Filtering provides a precision
improvement but at the cost of recall; total F1 is comparable. For temporal
relations specifically, the efficiency gains are the strongest result from RQ2.

---

## RQ3 — Rationale vs no rationale (B vs D)

| Metric | No rationale (B) | Rationale (D) | Change |
|---|---|---|---|
| Avg call time (s) | 1.50 | 6.65 | **+4.4× per call** |
| Time/doc (s) | 86 | 380 | **+4.4× total** |
| Precision | 0.2002 | 0.1958 | −0.0044 |
| Recall | 0.3669 | 0.3332 | −0.0337 |
| F1 | 0.2590 | 0.2466 | −0.0124 |

For temporal relations, adding rationale generation **does not improve
precision** — it marginally decreases both precision (−0.4%) and recall
(−3.4%), at a cost of **4.4× longer inference time per call**. This is
counter to the self-correction hypothesis in RQ3 for temporal relations.

---

## Comparison with baselines

These comparisons are directional only — different model (not fine-tuned),
different evaluation scale (N=20 docs vs full dataset), and for LLMERE,
different evaluation setting (see `docs/methodology.md`).

| System | Temporal P | Temporal R | Temporal F1 |
|---|---|---|---|
| RoBERTa [1] reproduced (15% train, valid) | 0.4470 | 0.5381 | 0.4884 |
| RoBERTa [1] paper | 0.578 | 0.539 | 0.558 |
| LLMERE [3] paper (Llama3-8b-instruct, fine tuned) | 50.5 | 59.9 | 54.8 |
| **Ours — Config D (full pipeline)** | 0.1958 | 0.3332 | 0.2466 |
| **Ours — Config B (batched+filtered)** | 0.2002 | 0.3669 | 0.2590 |
| **Ours — Config C (full-doc baseline)** | 0.1781 | 0.5460 | 0.2686 |

All our numbers are substantially below the reproduced RoBERTa baseline (F1
0.488 vs 0.259 best config). The gap is primarily attributable to: (1) no LLM
fine-tuning on the ERE task; (2) missing full document context in prompts;
(3) no candidate order preservation  (see `docs/methodology.md`, Implementation
Gaps); (4) retrieval quality. Closing this gap through fine-tuning, document context addition and improving retrieval quality is the target for M2.

---

## Notes

- Total n_predictions=20,282 for all configs — this is the full set of
  (query, candidate) pairs evaluated. With all_heads scope, every event in
  every document becomes a query.
- Config A wall time (649s/doc) highlights why single-pair classification is
  impractical at scale: 20,282 individual API calls for 20 documents.
- The avg per-call time for config A (0.64s) vs B (1.50s) and C (1.45s)
  shows that larger prompts (more candidates per call) take longer per call,
  but far fewer calls are needed total.