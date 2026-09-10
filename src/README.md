# Source Code

Main package: `ere_rag/`

- `data/`: data loading and preprocessing scripts
- `retriever/`: S-BERT fine-tuning and FAISS index construction  
- `llm/`: prompt design and inference (for intial experiments)
- `eci/`: prompt design and inference (for refined experiments)
- `evaluation/`: precision, recall, F1 computation and baseline scripts
- `baselines/`: reproduction of baseline experiments on the MAVEN-ERE dataset.
  - `temporal/`: baseline implementation and experiments for temporal relation extraction
  - `causal/`: baseline implementation and experiments for causal relation extraction