# Project Plan

## Topic

Retrieval-augmented Event Relation Extraction (ERE): a two-stage pipeline
combining a relation-type-aware dense retriever with partitioned batch LLM
classification and rationale generation, evaluated on MAVEN-ERE for causal
and temporal relation extraction.

## Motivation and gap

Event Relation Extraction — identifying temporal and causal relations
between events in text — underpins applications including knowledge graph
construction, question answering, and event prediction. Standard approaches
require pairwise comparison of all event mentions, resulting in quadratic
complexity that scales poorly as documents grow longer, and ERE at document
scale remains unsolved despite its importance.

The pairwise RoBERTa baseline [1] (temporal F1=55.8, causal F1=30.6,
temporal recall=53.9, causal recall=27.2, all-event-pairs setting)
establishes the quadratic-cost reference point. Efficient ERE [2] proposes
a retrieve-then-classify pipeline that improves F1 to 32.8 on *all
non-negative* pairs, but its bi-encoder retriever is fine-tuned for binary
relatedness only — it does not distinguish between relation types — and its
second stage is a cross-encoder that produces no reasoning trace. Any pair
dropped at retrieval is permanently unrecoverable. LLMERE [3] reformulates
ERE as question answering using a fine-tuned LLM, reducing model calls from
O(n²) to O(n) and achieving temporal F1=54.7 and causal F1=36.0 on all
event pairs. However, LLMERE partitions the *entire unfiltered document*
before prompting, exposing each partition to noise from unrelated events;
this noise problem is particularly acute for causal ERE, where KnowQA [4]
identifies hallucination — predicting causal relations where none exist —
as a critical failure mode. TacoERE [5] addresses information redundancy
through document clustering and summarization (compression-then-extraction,
causal F1=34.1 on MAVEN-ERE), and MAQInstruct [6] reduces inference samples
by reframing relation extraction as event selection given a relation type.
Neither combines retrieval with batched LLM reasoning.

To our knowledge, no existing work combines relation-type-aware dense
retrieval — a bi-encoder trained separately per relation type rather than
for generic binary relatedness — with partitioned batch LLM reasoning
applied exclusively to pre-filtered candidates, and rationale generation,
in a single pipeline for causal and temporal ERE.

**Key deviations from cited papers** (see `docs/methodology.md` for full
detail):
- This work does not fine-tune the LLM (LLMERE does). Compute constraints
  required a prompting-only approach on a hosted API; this is an explicit
  limitation.
- LLMERE derives rationales from structured dataset annotations (transitive
  chains from ground-truth event graphs). This work uses LLM-generated
  free-text rationales, which is closer to chain-of-thought prompting.


---

## Research Question

Does combining a relation-type-aware dense retriever with partitioned batch
LLM classification and rationale generation (method) improve causal and
temporal relation extraction precision, and reduce inference time per
document (metric), on MAVEN-ERE [1] (dataset), compared to exhaustive
pairwise enumeration, unfiltered full-document LLM partitioning, and
classification without rationale generation (comparison)?

---

### RQ1 — Retrieval effectiveness (recall vs. efficiency)

**Full question**: Can a relation-type-aware dense retriever effectively
replace exhaustive pairwise enumeration for causal and temporal event
relation candidate selection on MAVEN-ERE without significant loss in
recall?

**Hypothesis**: A dense retriever fine-tuned separately per relation type
achieves a genuine recall/efficiency tradeoff — meaning recall does not
collapse to near-zero at any practically useful percentage (which would
mean the retriever learned nothing useful), and the retriever does not
require returning nearly all document events to achieve decent recall
(which would mean it fails to filter). Concretely: at some percentage below 100%, retrieval recall approaches the RoBERTa baseline's
reported recall (53.9% temporal, 27.2% causal), using substantially fewer
candidates than all-pairs enumeration.

**Minimal experiment**: run the percentage sweep on one category with one
trained retriever. If recall never meaningfully exceeds a random baseline
at any percentage, or if decent recall requires >90% of all events, the
hypothesis is invalidated. 

---

### RQ2 — Batching and filtering (precision and efficiency)

**Full question**: Does presenting pre-filtered retrieved candidate event
pairs in partitioned batches to an LLM maintain or improve **precision**
and inference efficiency compared to full-document partitioning or
single-pair classification?

**Correction from Artefact 1**: the submitted research narrative for
Artefact 1 used the word "accuracy" in this research question. On
reflection, this was imprecise. Recall at the LLM stage is already bounded
by the retriever's recall ceiling — the LLM cannot recover relations the
retriever never presented to it. The LLM stage's primary job is to be
precise without leaving out too many positives: out of the candidate pairs it receives, how many of its positive predictions are correct? The relevant outcome metric is therefore
**precision** (and secondarily inference time), not accuracy (which would
conflate the retrieval and classification contributions). This document
uses precision throughout.

**Hypothesis**: presenting pre-filtered candidates in batches reduces
per-partition noise relative to LLMERE's unfiltered full-document
partitioning, improving precision without a proportional recall loss;
and batching achieves lower total inference time than single-pair
classification because fewer LLM calls are needed per document.

**Minimal experiment**: on a small set of documents, compare three
conditions (batched+filtered, batched+full-doc, single-pair+filtered)
with prompt and LLM held fixed. If batched+filtered shows no precision
improvement over batched+full-doc, the noise-reduction hypothesis is
not supported. If total time is not lower for batched vs. single-pair,
the efficiency hypothesis is not supported. 

---

### RQ3 — Rationale generation (precision)

**Full question**: Does requiring the LLM to generate a rationale for
each classified relation pair improve precision on causal and temporal
relations compared to classification-only approaches?

**Hypothesis**: requiring the LLM to generate a free-text rationale
improves precision specifically (not necessarily recall) relative to
classification-only prompting, because the rationale acts as a
self-correction mechanism — forcing the model to articulate why a
relation exists may reduce spurious positive predictions. Some cost in
inference time per call is expected.

**Minimal experiment**: on the same set of documents used for RQ2, run
the batched+filtered condition once with rationale-required prompt and
once without. If precision does not improve with rationale generation,
the self-correction hypothesis is not supported. 

---

## References

[1] Xiaozhi Wang et al. MAVEN-ERE: A unified large-scale dataset for event
coreference, temporal, causal, and subevent relation extraction. EMNLP
2022, pages 926–941.

[2] Ruochen Li, Zimu Wang, and Xinya Du. Efficient document-level event
relation extraction. RepL4NLP-2025, pages 92–99. ACL 2025.

[3] Zhilei Hu, Zixuan Li, Xiaolong Jin, Long Bai, Jiafeng Guo, and Xueqi
Cheng. Large language model-based event relation extraction with
rationales. COLING 2025, pages 7484–7496.

[4] Zimu Wang, Lei Xia, Wei Wang, and Xinya Du. Document-level causal
relation extraction with knowledge-guided binary question answering.
Findings of EMNLP 2024, pages 16944–16955.

[5] Yong Guan et al. TacoERE: Cluster-aware compression for event relation
extraction. LREC-COLING 2024, pages 15511–15521.

[6] Jun Xu, Mengshu Sun, Zhiqiang Zhang, and Jun Zhou. MAQInstruct:
Instruction-based unified event relation extraction. WWW Companion 2025,
pages 1441–1445.