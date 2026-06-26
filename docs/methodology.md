# Methodology

This document describes the experimental design of the ERE-RAG pipeline.
---

## Framing

This project proposes and evaluates a two-stage pipeline for Event Relation
Extraction (ERE): a dense retrieval stage that reduces the candidate event
pair space, followed by a batch LLM classification stage with optional
rationale generation. The work is not a reproduction of any single prior
system — it draws on design elements from Efficient ERE [2] (marker-token
retrieval) and LLMERE [3] (bucket-format prompting, partitioned batching,
rationale generation) and combines them in a way that neither paper does.
Comparisons with prior numbers are directional references, not controlled
head-to-head replications (different models, data slices, hardware, and in
LLMERE's case fine-tuning vs. prompting). See Threats to Validity.

---

## Datasets

**MAVEN-ERE** [1]: 4,480 English Wikipedia documents, 112,276 events,
1,216,217 temporal relations, 57,992 causal relations. Standard
train/valid/test split.
The test split carries no public labels — evaluation requires submission
to a CodaLab leaderboard.

**Evaluation protocol note.** MAVEN-ERE supports two settings used
inconsistently across papers and not directly comparable:
- *All event pairs* (harder): denominator includes NOREL pairs. Used by
  LLMERE and this work.
- *All non-negative event pairs* (easier): only annotated-positive pairs
  in denominator. Used by Efficient ERE.

All numbers in this project use the *all event pairs* setting, matching
the RoBERTa baseline [1] we reproduce. Efficient ERE's F1=32.8 is not
directly comparable to LLMERE's temporal F1=54.7 or our numbers for this
reason. We note that LLMERE's own evaluation script uses a further
re-split: they shuffle the official MAVEN-ERE train split 80/20 into
their own working train and valid, and treat the official MAVEN-ERE valid
split as their test set.
This means their "test" numbers are evaluated on the same official valid
split we use — so comparisons against their reported numbers are more
meaningful than they might appear.

**Splits used in this project:**
- **Train**: subsetted to **15% of documents** (random sample, fixed
  seed=42) for compute reasons. Used only to fine-tune the retriever.
- **Valid**: used **in full** (710 documents) for te RoBERTa baseline and retriever, partially used in the llm stage.
- **Test**: not used this milestone. No CodaLab submission was made; see
  Threats to Validity.

---

## Preprocessing

Event mentions are wrapped with `<m>...</m>` marker tokens before encoding
(adopted from Efficient ERE [2]), directing encoder attention to the target
event within its sentence context.

Two separate candidate universes per document, reflecting MAVEN-ERE's
annotation structure:
- **Temporal**: event mentions + TIMEX expressions. Temporal relations in
  MAVEN-ERE can reference either.
- **Causal**: event mentions only. `causal_relations` never reference
  TIMEX ids.

**Negative distribution for retriever data preparation and training.** For each anchor event,
all in-document events with no annotated relation to that anchor form the
negative pool. This pool is evenly distributed across the anchor's
positive pool, so no negative is wasted — an anchor with 5 positives
and 20 negatives gets 4 negatives assigned per positive pair.
For the traning we keep a fixed number k of negatives per positives depending on the category.

---

## Baselines

1. **Pairwise RoBERTa** — reproduced from MAVEN-ERE's own released
   baseline code, evaluated on the **full** valid split, training seed=42,
   training data subsetted to 15%.
   Single seed only — see Threats to Validity.

2. **Efficient ERE** [2], **LLMERE** [3] — reprocuded with a couple of differences worth noting.
   Citing reported numbers from prior work is standard practice when
   reproducing their full setup is infeasible (different hardware, model,
   data slice). Differences in evaluation setting and data are noted
   explicitly wherever comparisons are made.

---

## Model Variants

### Retriever

**Architecture**: `all-MiniLM-L6-v2` (S-BERT family, sentence-transformers)
fine-tuned with `TripletLoss` (cosine distance, margin=0.5).

**Trained separately per relation category.** This is a deliberate design
choice — not a deviation from prior work. Efficient ERE's retriever is
trained for binary relatedness (any relation vs. none) without
relation-type specificity. Training separate temporal and causal retrievers
is the central mechanism tested in RQ1: Can a relation-type-aware dense retriever effectively replace exhaustive pairwise enumeration for causal and temporal event relation candidate selection on MAVEN-ERE without significant loss in recall?

**Negative sampling strategy.** The full negative pool per anchor is
distributed evenly across that anchor's positives at data-prep time
(see Preprocessing above). At training time, k negatives are sampled
**once per run** (seeded by `--seed`) and reused for all epochs. This
differs from per-epoch resampling (which would expose the model to more
negative variety but makes runs less reproducible) and from using all
available negatives (which would severely inflate training set size,
especially for causal).

**Choice of k (negatives per positive pair).** Informed by dataset
statistics computed on the train split only using
`src/ere_rag/data/temporal_stats.py` and `causal_stats.py` (see
`src/ere_rag/data/stats_README.md` for full interpretation):

- **Temporal k=3**: average out-degree = 8.69 positives/event, average
  non-related events = 22.85/event. Ratio ≈ 2.63 → ceiling integer = 3.
  Setting k=3 means each positive pair receives roughly as many negatives
  as the average ratio of negatives to positives per anchor, without
  requiring sampling with replacement for most anchors.
- **Causal k=5**: average out-degree = 0.53 positives/event, average
  non-related = 31.0/event. The raw ratio (~58) is impractically large
  and reflects extreme sparsity (most events have zero causal relations).
  k=5 is a practical upper bound chosen to provide harder training signal
  for the causal retriever without overloading it with too many negatives.

**Max training groups** (due to compute constraints): 10,000 for temporal,
5,000 for causal (causal has far fewer available groups total — most event
pairs have no annotated causal relation, so anchors with at least one
positive are not as common as in the temporal split).

**Training seeds**: Used for sampling groups and negatives: {42, 1, 0} × 2 categories = 6 runs. Satisfies the
≥3-seed requirement for variance reporting.

**`--category both`** (union of temporal and causal positives): implemented
but not tested this milestone — following LLMERE's approach
of treating temporal and causal as separate tasks. May be added later on.

---

### LLM Stage

**Retrieval cutoff**: percentage-based rather than fixed-k, because fixed-k
is document-size-blind — k=5 is proportionally too small for a 50-event
document and proportionally too large for a 10-event document. Using
percentage of `n−1` (other events in doc) keeps the retrieval budget
proportional to document size. The full sweep (10/20/30/50%) is evaluated
for RQ1. A separate operating point (60% temporal / 50% causal) is chosen
for RQ2/RQ3 — see Evaluation Protocol.

**Output format**: LLMERE-style relation-subtype buckets, e.g.:
```
CAUSE: <e0>, <e2>
PRECONDITION: none
```

**Rationale format**: free-text natural language explanation generated
inline with each prediction. **This differs from LLMERE's rationales**,
which are structured dataset-derived annotations (transitive chains from
ground-truth event graphs and coreference information that require access
to labeled data to construct and fine tuning). Our rationales are generated by the LLM at
inference time and are closer to chain-of-thought prompting — they test
whether the act of generating an explanation improves prediction precision,
independent of whether that explanation is grounded in structured knowledge.

**Model**: Qwen3-Next-80B-A3B-Instruct, accessed via a hosted institutional
API. See Implementation Choices below for the full justification.

**Temperature**: 0.0 (deterministic decoding).

**query_scope**: `all_heads` — every event in a document is used as a query,
not only those with known positive relations. An earlier `positive_heads`
mode exists in the code for faster piloting but is not used for any
reported result, since it silently drops zero-positive queries and biases
precision upward.

---

## Evaluation Protocol

### Retriever (RQ1)

**Percentage-based retrieval sweep** across [5, 10, 20, 30, 50, 60]% of `n−1`
other events per document. For a document with n events and cutoff p%,
the retriever returns the top `round(p × (n−1))` nearest neighbours
(FAISS `IndexFlatIP`, cosine similarity via normalised embeddings, ≥1).
Metrics: micro-averaged precision, recall, F1 across all query events with
at least one annotated positive.

A fixed-k evaluation (comparing specific values of k against the
percentage-based approach) is planned as a supplementary analysis — the
cached indices and embeddings are stored, so this does not require
rerunning `build_index.py`.

### LLM Stage (RQ2/RQ3)

**Why two operating points?** RQ1 reports the full sweep to characterise
the retrieval tradeoff. RQ2/RQ3 need a single operating point for
controlled ablation. This point is chosen to maximise recall (so the LLM
sees as many true positive candidates as possible) while providing at least
some filtering over full-document enumeration. 60% temporal / 50% causal
give retrieval recall ≈ 0.69 — choosing not to lose
more than ~30% of true positive candidates before the LLM stage.


**RQ2 experiment design** — three conditions per relation type, holding
prompt fixed (no rationale, few-shot):
1. Batched + filtered: 60%/50% cutoff, partition size 25/20(temporal/causal)
2. Single-pair + filtered: 60%/50% cutoff, partition size 1
3. Batched + full document: 100% cutoff, partition size 25/20

**RQ3 experiment design** — two conditions, holding retrieval and partition
fixed at the batched+filtered operating point:
1. Rationale ON: `llmere_few_shot`
2. Rationale OFF: `llmere_few_shot_no_rationale`

Condition 1 of RQ2 (batched+filtered, no rationale) doubles as the
rationale-OFF arm of RQ3 — the same run serves both tables.


All LLM experiments: N=20 documents, valid split, query_scope=all_heads,
seed=42 retriever model.

---

## Implementation Choices vs. Cited Methods

### Deliberate design choices (these are contributions, not deviations)

- **Separate retriever per relation type**: Efficient ERE's retriever is
  trained for binary relatedness (related vs. not), with no relation-type
  signal. Training separate temporal and causal retrievers is what RQ1
  tests. This also happens to align with LLMERE's separate handling of
  relation types, but was motivated by the theoretical argument that
  relation-type-specific contrastive training should produce better-ranked
  candidates.

- **Pre-filtered partitioning**: LLMERE applies its partitioning strategy
  to the full document (all events partitioned into groups). This work
  applies partitioning only to pre-filtered retrieved candidates, so each
  partition contains only events the retriever considers likely to be
  related. This is the mechanism tested in RQ2.

- **Percentage-based retrieval**: instead of Efficient ERE's fixed k,
  retrieval cutoff scales with document size (see Evaluation Protocol).

### Genuine implementation gaps (limitations, not contributions)

- **Document context not included**: LLMERE feeds the full document into
  every prompt, with all events marked by sequential ids. This pipeline
  provides only per-event sentence snippets from the retrieval cache.
  Full-document context was omitted due to two compounding constraints:
  (1) early experiments used Groq API which enforced tight tokens-per-minute
  limits, forcing shorter prompts; (2) the retriever cache does not store
  full document text. A hook for document context exists in `prompt.py`
  but is not populated. This likely explains part of any performance gap
  versus LLMERE's reported numbers. Adding full document context is a
  planned improvement.

- **Candidate order not preserved**: LLMERE includes the full document so
  the LLM sees events in their original document order. Our retrieval stage
  reorders candidates by embedding similarity, so candidates presented to
  the LLM may not appear in their original narrative order. This could
  introduce a false-order bias — the model may interpret earlier positions
  in the candidate list as earlier temporal positions in the document, when
  they are actually highest-similarity positions. This is an uncontrolled
  confound worth acknowledging.

- **LLM model differs from LLMERE**: LLMERE fine-tunes Llama3-8B via LoRA.
  We use prompting (zero-shot / few-shot) with Qwen3-Next-80B-A3B-Instruct
  via hosted API and also for now the few-shot part is only to ensure the model respects the required output format. Full model substitution justification below.

- **No LLM fine-tuning**: LLMERE fine-tunes the LLM end-to-end on the
  ERE task. This work uses only prompting. This means our approach tests
  whether the pipeline design (filtering + batching + rationale) provides
  gains in the more accessible prompting-only scenario, which is not the
  same question LLMERE answers.

- **Prompt engineering**: some prompt versions were piloted. The final chosen configuration (`llmere_few_shot`
  / `llmere_few_shot_no_rationale`) was chosen based on small-N piloting
  and represents a reasonable but not exhaustively tuned starting point.
  Prompt sensitivity analysis is out of scope for this milestone.

---

## LLM Model Choice — Full Justification

**Why not Llama3-8B-base (LLMERE's model)?**
The model weights total ~16GB. The available hardware (Apple M1, 8GB
unified memory) cannot load this model locally. Attempted approaches in
order:
1. Local loading via HuggingFace Transformers → out-of-memory at load time.
2. Groq API with `llama-3.1-8b-instant` → but hard
   daily token limits and tokens-per-minute rate limits caused repeated
   pipeline crashes after 3–4 documents with even with the shorter prompt versions and small partition sizes.
3. Institutional hosted API → selected Qwen3-Next-80B-A3B-Instruct.

**Implication for comparisons**: any performance difference from LLMERE's
reported numbers cannot be cleanly attributed to the pipeline design alone,
since the model differs. Comparisons are directional only. This is stated
explicitly as a threat to validity.

**Wall-clock time vs. CPU time**: timing reported in this project is
wall-clock time (real elapsed time including API network latency, not just
local CPU cycles). "Orchestration" cost (prompt building, response parsing,
file I/O) is negligible relative to API round-trip time. LLMERE's reported
times (14.4s/doc batched, 90.3s/doc pairwise) are on an A100 GPU with 40GB memory with a
locally running model — not comparable to API-based wall-clock times.
Within-project comparisons (batched vs. single-pair call count and
per-doc time) are directly interpretable even though cross-paper timing is
not.

---

## Metrics

- **Primary**: micro-averaged precision, recall, F1 per relation type
  (temporal, causal) following Wang et al. [1].
- **Secondary**: number of LLM calls per document, average wall-clock time
  per document. These are compared *within* the project (across RQ2
  conditions) and used only directionally against LLMERE's reported numbers
  given the hardware/model differences noted above.

---

## Random Seeds

- **Data sampling seed**: 42, fixed everywhere. Used for: document
  subsetting (15% of train), data preparation for the retriever. 
- **Retriever training seed**: varied across {42, 1, 0}. The training seed
  controls: group sampling and negative sampling at dataset construction.
- **LLM stage**: 42 to sample N documents for inference (fixed for now).
- **RoBERTa baseline**: seed=42 only (single run). See Threats to Validity.

---

## Compute Budget

- **Hardware**: Apple M1 (8GB unified memory). All retriever training and
  FAISS operations run on CPU (MPS backend attempted; produced
  out-of-memory errors for batch sizes ≥16).

- **LLM stage**: hosted API. Local cost is orchestration only.

---

## Threats to Validity

- **Model substitution** (computational budget / construct validity):
  Qwen3-Next-80B-A3B-Instruct replaces LLMERE's fine-tuned Llama3-8B
  due to hardware constraints (see LLM Model Choice above). The model is
  larger, uses a different architecture, and is not fine-tuned on the ERE
  task. Any performance difference from LLMERE's numbers conflates pipeline
  effects with model capability effects.

- **No LLM fine-tuning** (construct validity): testing the pipeline in a
  prompting-only setting answers a different question than LLMERE's
  fine-tuning setup. Results from this project characterise the pipeline's
  value in the prompting regime specifically.

- **Evaluation split** (evaluation protocol): all results on valid split
  only. The MAVEN-ERE test split has no public labels and submission to
  CodaLab was not attempted this milestone due to compute constraints
  (LLM experiments run on N=20 documents, not the full valid set let alone
  the test set). The RoBERTa baseline is evaluated identically (valid only)
  for internal consistency. We note that LLMERE also uses the official
  MAVEN-ERE valid split as their de-facto test set, so this alignment
  makes comparisons more meaningful than they might initially appear.

- **Small LLM evaluation sample** (statistical power): N=20 documents for
  RQ2/RQ3 experiments. Predictions within a document are not independent
  (events in the same document share context). Results should be treated
  as preliminary / indicative.

- **Single retriever seed for LLM experiments** (reproducibility):
  LLM-stage experiments use only the seed=42 retriever model. Seed
  sensitivity of the retriever (seen in the ±0.047 std on triplet accuracy)
  is not propagated through to the LLM-stage results.

- **Dataset scope** (external validity): training on 15% of MAVEN-ERE
  train (seed=42) so only 1 seed for now.

- **Candidate order not preserved** (construct validity): see Implementation
  Gaps above.

- **Document context absent** (construct validity): see Implementation Gaps
  above.

- **Prompt sensitivity** (construct validity): final prompt versions are
  reasonable but not exhaustively optimised. Results may improve with
  better prompt engineering.
