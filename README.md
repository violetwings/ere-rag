# ere_rag: Retrieval-Augmented Event Relation Extraction

Master's thesis project — University of Passau, 2026.

This project investigates whether dense retrieval can replace exhaustive 
pairwise enumeration in event relation extraction, combining a fine-tuned 
S-BERT retriever with partitioned batch LLM reasoning for causal and 
temporal relation extraction on MAVEN-ERE.

See `docs/project-plan.md` for the full research plan and hypothesis.

The template supports a research loop like the one presented in "Scientific Working I — The Research Loop":

1. read literature and collect evidence
2. identify a gap
3. define a research question
4. state a falsifiable hypothesis
5. design a minimal experiment
6. run pre-experiments and sanity checks
7. run full experiments systematically
8. document results and limitations
9. communicate outcomes in reports and presentations

Git is not just storage here. It is the lab notebook. GitLab is not just hosting. It is the project management layer for research questions, experiments, milestones, and review.

## What This Template Is For

Use this template when you want one repository to manage:

- source code in `src/`
- project documentation in `docs/`
- final reports in `docs/reports/`
- experiment runs and configurations in `experiments/`
- notebooks for exploration in `notebooks/`
- reproducibility metadata such as environment files and changelog

This layout works well for:

- seminar projects
- thesis prototypes
- exploratory ML / IR / NLP research
- experimental software engineering
- reproducible baseline and ablation studies

## Repository Structure

```text
.
├── .gitignore
├── .gitlab/
│   └── issue_templates/
│       ├── Experiment.md
│       └── Research Question.md
├── CHANGELOG.md
├── LICENSE
├── README.md
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── reports/
│   │   └── README.md
│   ├── methodology.md
│   ├── project-plan.md
│   └── README.md
├── experiments/
│   └── README.md
├── notebooks/
│   └── README.md
├── pyproject.toml
└── src/
    ├── README.md
    └── project_name/
        └── __init__.py
```

## Directory Guide

### `src/`

Production-quality project code lives here. Keep reusable logic in modules, not in notebooks. A good rule is:

- code that must be rerun belongs in `src/`
- one-off exploration may start in notebooks, then be moved into `src/`

### `docs/`

Working documentation for the project:

- literature notes and references
- research question and hypothesis
- experiment design
- methodology decisions
- limitations and threats to validity

This folder should explain why the project is being done, not just how to run code.

### `docs/reports/`

Final report artifacts belong here:

- seminar report
- thesis-style writeups
- poster source
- presentation notes

Keep submitted or near-submission outputs separate from working notes.

### `experiments/`

This is where experiment runs are tracked. The intended pattern is:

- one experiment = one folder
- each folder contains at least `README.md`, `config.yaml`, and `results.json`
- all runs are logged, including failed or non-reportable ones

Suggested naming:

```text
experiments/
└── 2026-04-21_bm25_baseline/
    ├── README.md
    ├── config.yaml
    └── results.json
```

This follows the lecture guidance to work in small loops first, then scale only after sanity checks pass.

### `notebooks/`

Use notebooks for:

- initial exploration
- plotting
- quick data inspection
- trying out ideas before hardening them into scripts

Do not let notebooks become the only source of truth for the pipeline.

### `data/`

Suggested semantics:

- `data/raw/`: original immutable inputs
- `data/interim/`: temporary transformed data
- `data/processed/`: clean model-ready data
- `data/external/`: third-party resources or downloads

Large datasets should usually not be committed. Use `.gitignore`, object storage, Git LFS, or a documented external source.

## Recommended Workflow

This template is structured around a practical research workflow.

### 1. Start with a research question

Create a GitLab issue using the `Research Question` template for each major research question or hypothesis track.

Each issue should define:

- the gap or motivation
- the exact research question
- the hypothesis
- the method, dataset, metric, and comparison
- success and failure criteria

If the question is vague, the project is not ready for full experimentation.

### 2. Plan with GitLab milestones

Use GitLab milestones to group work into meaningful checkpoints. A milestone should correspond to a research phase, not an arbitrary date bucket.

Good milestone examples:

- `M1 Literature Review and Gap`
- `M2 Baseline and Sanity Checks`
- `M3 Pre-Experiment`
- `M4 Full Experiment`
- `M5 Report Draft`
- `M6 Final Submission`

Each milestone should end in a verifiable outcome such as a baseline reproduced, a pre-experiment passed, or a report draft completed.

### 3. Use issues for concrete work

Use GitLab issues in two distinct ways:

- research-question issues: one issue for each substantial question or hypothesis
- experiment issues: one issue for each experiment, baseline, ablation, or evaluation run set

Recommended labels:

- `research-question`
- `experiment`
- `baseline`
- `ablation`
- `bug`
- `documentation`
- `report`
- `blocked`

Link experiment issues back to the research-question issue they serve.

### 4. Work in small loops before scaling

Follow the lecture's experimental discipline:

- implement a minimal baseline first
- verify shapes, inputs, outputs, and metrics
- overfit a tiny sample if relevant
- run a pre-experiment before a full run
- commit every stable milestone

The point is to fail early and cheaply.

### 5. Log every experiment

Every meaningful run should leave a trace in `experiments/`.

At minimum store:

- purpose of the run
- code version or commit hash
- configuration
- random seed(s)
- metric outputs
- notes on interpretation

Do not keep only the runs that look good enough to report.

### 6. Write while you work

Do not postpone documentation until the end. Update `docs/` throughout the project:

- revise the research question when it sharpens
- document methodology changes
- note threats to validity
- record why certain ideas were dropped

The final report becomes much easier when the reasoning is already captured.

## Git and Commit Practices

Treat commit history as a research record.

Recommended pattern:

- one coherent change per commit
- commit after each verified milestone
- write messages that describe both the change and the result when relevant

Examples:

```text
feat: add BM25 baseline evaluation pipeline
fix: correct nDCG@10 computation for multi-label relevance
docs: define hypothesis and primary metric in project plan
exp: log pre-experiment for dense retriever with 3 seeds
```

### Commit Message Behavior

Commit messages should behave like concise lab notebook entries.

- say what changed
- say why it changed when that is not obvious
- mention the result or finding if the commit captures a verified outcome
- keep the subject line short and specific
- use the body for metrics, observations, caveats, or next steps

Suggested format:

```text
<type>: <specific change>

Context: why this change was needed.
Result: what was observed, verified, or still uncertain.
Next: the next planned research or engineering step.
```

Example:

```text
exp: add BM25 baseline on MSMARCO dev

Context: needed a minimal lexical baseline before dense retrieval work.
Result: MRR@10 = 0.187 on dev, consistent with expectation from prior work.
Next: run a 3-seed pre-experiment for the dense retriever.
```

When a result matters, mention it in the body of the commit or merge request.

## Merge Requests

Use merge requests even for small research teams because they create review points.

A good merge request should answer:

- what changed
- why it changed
- what research question or issue it supports
- what evidence was produced
- what remains uncertain

If a merge request changes evaluation logic, reviewers should verify metric correctness, not only code style.

## Reproducibility Expectations

This template assumes the following discipline:

- dependencies are pinned
- seeds are fixed where randomness matters
- preprocessing is scripted
- results are tied to configuration
- reports cite the exact experiment conditions used

If someone else clones the repository, they should be able to understand:

- what was asked
- what was tested
- how it was tested
- what was concluded

## Suggested First Steps After Creating a New Project

1. Rename `project_name` under `src/` to the real package name.
2. Update the license if your institution or team requires a different one.
3. Fill in `docs/project-plan.md` with your topic, research question, and hypothesis.
4. Create GitLab labels and milestones.
5. Open one `Research Question` issue and one `Experiment` issue.
6. Add environment or dependency details to `pyproject.toml`.

## Minimal GitLab Setup

After creating the GitLab project, set up:

- labels for research and experiment tracking
- milestones for major research phases
- protected default branch if the team is larger than one person
- merge request templates if your group needs formal review
- optional CI jobs for tests, linting, or report builds

## Files Included in This Template

### `CHANGELOG.md`

High-level project history. This is not a replacement for git history. Use it for noteworthy milestones and releases.

### `LICENSE`

Default license for the template. Replace it if your institution, lab, or client requires a different one.

### `.gitignore`

Preconfigured for Python, notebooks, virtual environments, experiment artifacts, and common data-science clutter. Adjust it to your stack.

## What Good Practice Looks Like In This Template

Good:

- `docs/project-plan.md` states the hypothesis before full experiments
- `experiments/2026-04-21_bm25_baseline/` contains config and results
- a GitLab issue links the experiment to a research question
- a milestone groups baseline, pre-experiment, and report checkpoints

Bad:

- research question exists only in chat or email
- results are kept only in screenshots
- code differs from what the report describes
- only successful runs are documented
- notebooks contain critical preprocessing that is nowhere scripted

## Maintenance

This repository is a template stub. It is intentionally lightweight, but it should not stay generic for long. The first project using it should replace placeholders with actual project names, datasets, metrics, milestones, and experiment records.

## License

This template ships with the MIT License. Replace it if needed.

## Setup

### 1. Clone the repository

```bash
git clone https://git.fim.uni-passau.de/nourellil/ere_rag.git
cd ere_rag
```

### 2. Create and activate virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install LLaMA-Factory from source

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
cd ..
```

### 5. Download MAVEN-ERE

Download the dataset from https://github.com/THU-KEG/MAVEN-ERE 
and place the files in `data/raw/`.

## Data

MAVEN-ERE is the primary benchmark. Due to compute constraints, 
experiments are conducted on a fixed 15% random-seed subset 
(seed=42) of the full dataset.

## Reproducing Results

### Step 1 — Preprocess the data
```bash
python src/ere_rag/data/preprocess.py
```

### Step 2 — Run the RoBERTa baseline
```bash
python src/ere_rag/evaluate/baseline_roberta.py
```

### Step 3 — Fine-tune the S-BERT retriever
```bash
python src/ere_rag/retriever/train_retriever.py
```

### Step 4 — Build the FAISS index
```bash
python src/ere_rag/retriever/build_index.py
```

### Step 5 — Run the full pipeline
```bash
python src/ere_rag/llm/inference.py
```

## Requirements

- Python 3.10
- See requirements.txt for full dependency list
- GPU recommended for LLM inference (Llama3-8B)

## References

- MAVEN-ERE: Wang et al., EMNLP 2022
- Efficient ERE: Liu et al., COLING 2025
- LLMERE: Wei et al., ACL 2025