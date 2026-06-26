# Data Statistics — `src/ere_rag/data/`

## Files

- **`temporal_stats.py`** — computes per-event temporal relation statistics
  over a MAVEN-ERE split. Builds a directed graph from `temporal_relations`
  (with SIMULTANEOUS and BEGINS-ON symmetrically expanded) and reports
  degree statistics per event.
- **`causal_stats.py`** — same structure for `causal_relations`. No symmetric
  expansion — CAUSE and PRECONDITION are directional.

## Usage

Always run on the **train split only**. Using the valid split here would
constitute data leakage: any choice of k or retrieval percentage informed
by validation statistics would be tuned to those documents, not held out.

```bash
python src/ere_rag/data/temporal_stats.py --data_path data/raw/MAVEN_ERE/train.jsonl
python src/ere_rag/data/causal_stats.py   --data_path data/raw/MAVEN_ERE/train.jsonl
```

## Output fields — what each means

For each event in each document, the scripts count:
- `out_degree`: how many other events it has a **positive** relation to
  (i.e., the event appears as the "head" in an annotated relation pair)
- `not_out`: how many events in the same document it does **not** have a
  positive relation to, i.e., `n − 1 − out_degree` where n = doc event count
- `norm_degree`: `out_degree / (n − 1)` — the fraction of the document's
  other events that are positively related to this one

Reported statistics aggregate these across all events in all documents:

| Field | What it means |
|---|---|
| `avg_out` | Average positive relations per event (anchor's positive pool size) |
| `avg_not_out` | Average non-related events per event (anchor's negative pool size) |
| `avg_norm_degree` | Average fraction of doc events positively related to an anchor |
| `avg_events` | Average events per document |

## Results on MAVEN-ERE train split

```
Temporal:
  avg_out:         8.69    (avg positives per event)
  avg_not_out:    22.85    (avg negatives available per event in same doc)
  avg_norm_degree: 0.292   (avg ~29% of doc events are positive for a given anchor)
  avg_events:     23.34    (avg events per document)

Causal:
  avg_out:         0.53    (avg positives per event — very sparse)
  avg_not_out:    31.00    (avg negatives available per event in same doc)
  avg_norm_degree: 0.025   (avg ~2.5% of doc events are causally related to an anchor)
```

## How these numbers informed k (negatives per positive pair) for retriever training

The retriever is trained on triplets (anchor, positive, negative). For each
(anchor, positive) pair we assign k negatives. The ratio `avg_not_out /
avg_out` gives an upper bound on how many negatives are available per
positive, averaged across the corpus:

**Temporal**: `22.85 / 8.69 ≈ 2.63` → ceiling = **3**. k=3 means each
positive pair sees at most 3 negatives, matching the average availability
without wasting negatives (more negatives per positive than this would
require sampling with replacement for most anchors).

**Causal**: `31.00 / 0.53 ≈ 58.5` — far too many to use all. The ratio is
driven by extreme sparsity (most events have zero causal relations). k=5 was
chosen as a practical upper bound that acknowledges the severe positive/
negative imbalance without overloading training with redundant negatives.
In practice, 5 negatives per causal positive is still a large ratio; the
choice reflects that causal retrieval needs harder training signal than
temporal to learn any meaningful signal at all.

## How avg_norm_degree informed retrieval percentages for RQ1

At retrieval time, for a document with n events, retrieving a percentage p
of `n − 1` other events gives roughly `p / avg_norm_degree` times more
candidates than true positives. The `avg_norm_degree` values (0.292
temporal, 0.025 causal) serve as reference points: a retriever using
percentage ≥ avg_norm_degree as its cutoff should in principle recall the
majority of true positives if ranking is meaningful, while a retriever
requiring a much higher percentage just to reach reasonable recall has not
learned useful signal. The full RQ1 sweep (10/20/30/50%) uses these as
anchors — see `experiments/retriever/temporal/README.md` and
`experiments/retriever/causal/README.md` for results and analysis.