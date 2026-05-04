# Concept-kNN

**Interpretable Graph Clustering of LLM Conversations via Classical NLP**

[![Paper](https://img.shields.io/badge/IEEE-CONECCT%202026-blue)](https://ieeexplore.ieee.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Concept-kNN is a scalable, taxonomy-free clustering pipeline for organizing large-scale LLM conversations. It uses classical NLP techniques — dependency parsing, named entity recognition, and noun phrase chunking — to extract interpretable concept vectors, enriches them via PMI densification, and clusters the resulting k-nearest neighbor graph using Leiden community detection with cascading singleton reassignment.

> **TL;DR:** Outperforms 13 baselines (including ClusterLLM, TnT-LLM, Clio) by 7–21% on two large-scale datasets at zero API cost.

---

## Key Results

| Method | ShareGPT (162K) | LMSYS (98K) | API Cost |
|:-------|:---:|:---:|:---:|
| **Concept-kNN (Ours)** | **0.627** | **0.635** | **$0** |
| SBERT + UMAP + KMeans | 0.568 | 0.591 | $0 |
| SBERT + Leiden | 0.561 | 0.565 | $0 |
| ClusterLLM | 0.536 | 0.554 | ~$0.03 |
| Clio | 0.516 | 0.526 | ~$23 |
| TnT-LLM | 0.131 | 0.139 | ~$3.50 |

*Balanced score = NMI_fine × coverage. Full results with 13 baselines in the paper.*

---

## Pipeline Architecture

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐
│  Stage 1 │───▶│   Stage 2    │───▶│  Stage 3  │───▶│   Stage 4    │
│ NLP      │    │ Filter &     │    │ PMI       │    │ Hybrid       │
│ Extract  │    │ Weight       │    │ Densify   │    │ Vectors      │
│          │    │              │    │           │    │              │
│ spaCy    │    │ IDF filter   │    │ top-K PMI │    │ [α·vc;       │
│ NER, dep │    │ prompt boost │    │ associates│    │  (1-α)·vt]   │
│ parse,   │    │ conv.        │    │ per       │    │ L2-norm      │
│ regex,NP │    │ propagate    │    │ concept   │    │              │
└──────────┘    └──────────────┘    └───────────┘    └──────┬───────┘
                                                           │
    ┌──────────┐    ┌──────────┐    ┌───────────┐          │
    │  Output  │◀───│ Stage 7  │◀───│  Stage 6  │◀───┌─────┴──────┐
    │          │    │ Cascade  │    │  Leiden    │    │  Stage 5   │
    │  100%    │    │          │    │           │    │ kNN Graph  │
    │ coverage │    │ 6 rounds │    │ community │    │            │
    │ clusters │    │ τ=0.5→0  │    │ detection │    │ k=30,      │
    │          │    │ reassign │    │ γ         │    │ threshold σ│
    └──────────┘    └──────────┘    └───────────┘    └────────────┘
```

---

## Installation

```bash
git clone https://github.com/dnbresearch/concept-knn.git
cd concept-knn
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Requirements

- Python ≥ 3.10
- spaCy 3.8+ with `en_core_web_sm`
- scikit-learn ≥ 1.7
- igraph ≥ 1.0
- leidenalg ≥ 0.11
- sentence-transformers ≥ 3.3 (for baselines only)
- PyTorch ≥ 2.1 with CUDA (for SBERT baselines only)

---

## Quick Start

### 1. Prepare data

```bash
# ShareGPT
python scripts/prepare_data.py \
    --input data/sharegpt_raw.json \
    --output data/sharegpt_prepared.json

# LMSYS
python scripts/prepare_data.py \
    --input data/lmsys_raw.json \
    --output data/lmsys_prepared.json \
    --clean-lmsys  # removes system prompt leakage & bot spam
```

### 2. Extract features (NLP concepts + TF-IDF)

```bash
python scripts/feature_extraction_v5.py \
    --input data/sharegpt_prepared.json \
    --output data/sharegpt_features.json
```

### 3. Generate ground truth (requires OpenAI API key)

```bash
export OPENAI_API_KEY=<your-key>
python scripts/llm_ground_truth.py \
    --input data/sharegpt_features.json \
    --output ground_truth/ \
    --model gpt-4o \
    --granularity coarse mid fine
```

### 4. Run Concept-kNN

```bash
python scripts/concept_knn_v3.py \
    --input data/sharegpt_features.json \
    --gt ground_truth/gt_coarse_prompt_labels.json \
    --alpha 0.8 \
    --sigma 0.6 \
    --pmi-k 35 \
    --resolution 30 \
    --output results/concept_knn_results.json
```

### 5. Run baselines

```bash
# All non-LLM baselines
python scripts/baselines.py \
    --input data/sharegpt_features.json \
    --gt ground_truth/ \
    --output results/baselines/

# LLM-assisted baselines (requires API keys)
python scripts/baseline_clusterllm.py --input data/sharegpt_features.json
python scripts/baseline_tntllm.py --input data/sharegpt_features.json
python scripts/baseline_clio.py --input data/sharegpt_features.json
```

### 6. Ablation study

```bash
python scripts/ablation.py \
    --input data/sharegpt_features.json \
    --gt ground_truth/ \
    --output results/ablation/
```

---

## Hyperparameters

| Parameter | Symbol | Default | Description |
|:----------|:------:|:-------:|:------------|
| Blend ratio | α | 0.8 | Weight of concept vectors vs TF-IDF |
| Similarity threshold | σ | 0.6 | Minimum edge weight in kNN graph |
| PMI top-K | K | 35 | Number of PMI associates per concept |
| Resolution | γ | 30 | Leiden resolution parameter |
| kNN neighbors | k | 30 | Neighborhood size for graph construction |
| Cascade thresholds | τ | 0.5→0.0 | 6 rounds at [0.5, 0.4, 0.3, 0.2, 0.1, 0.0] |

All hyperparameters transfer across datasets (α, σ, K identical; only γ differs slightly: 80 vs 50). Performance varies by <1% across wide parameter ranges.

---

## Ablation Results

The improvement from TF-IDF+KMeans (0.487) to Concept-kNN (0.627) decomposes as:

| Component | Contribution | Percentage |
|:----------|:---:|:---:|
| Concept representation | +0.068 | 49% |
| Graph-based clustering (Leiden) | +0.059 | 42% |
| Hybrid blending (α=0.8) | +0.013 | 9% |

**Key finding:** Leiden amplifies signal in semantic representations (+10.7% on concepts, +14.3% on SBERT) but degrades on lexical features (−0.7% on TF-IDF).

---

## Ground Truth Validation

Three-layer validation of LLM-generated ground truth:

| Validation | Result | What it shows |
|:-----------|:------:|:--------------|
| LLM cross-model (Claude vs GPT) | κ ≈ 0.35 | Consistent across datasets |
| Circular bias test | τ = 1.0 | Rankings preserved, 1.8% NMI diff |
| Human annotation (3 annotators) | κ = 0.81 | Categories well-defined; LLM κ reflects propagation noise |

---

## Project Structure

```
concept-knn/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   └── default.yaml            # Default hyperparameters
├── scripts/
│   ├── prepare_data.py         # Data loading and cleaning
│   ├── feature_extraction_v5.py # NLP concept extraction
│   ├── concept_knn_v3.py       # Main Concept-kNN pipeline
│   ├── baselines.py            # Non-LLM baselines (KMeans, HDBSCAN, etc.)
│   ├── baseline_clusterllm.py  # ClusterLLM baseline
│   ├── baseline_tntllm.py      # TnT-LLM baseline
│   ├── baseline_clio.py        # Clio baseline
│   ├── ablation.py             # Representation × algorithm ablation
│   ├── cascade_analysis.py     # Cascade round-by-round analysis
│   ├── llm_ground_truth.py     # GT generation via LLM
│   ├── gt_validation.py        # Cross-model GT validation
│   ├── circular_bias_test.py   # Circular bias test
│   ├── sbert_encode.py         # SBERT embedding generation
│   └── eval_shared.py          # Shared evaluation utilities (NMI, coverage)
├── data/                       # Raw and processed datasets (not included)
├── ground_truth/               # Generated GT labels (not included)
├── results/                    # Experimental results
└── paper/                      # LaTeX source and figures
    ├── concept_knn_paper.tex
    └── hyperparam_sensitivity.pdf
```

---

## Datasets

| Dataset | Size | Source | Download |
|:--------|:----:|:------:|:--------:|
| ShareGPT | 162K conversations | Shared ChatGPT logs | [Link](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| LMSYS-Chat-1M | 98K conversations* | Chatbot Arena | [Link](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) |

*After cleaning: 97,776 items from 100K English first-turn sample.

---

## Runtime

All clustering benchmarks on a single machine (CPU only, 98K items):

| Method | Time | Slowdown |
|:-------|:----:|:--------:|
| **Concept-kNN** | **200s** | **1×** |
| SBERT + Leiden | 960s | 5× |
| SBERT + UMAP + KMeans | 1,245s | 6× |
| Concept + KMeans | 9,693s | 48× |

Concept extraction: ~3h 15m (ShareGPT), ~3h 54m (LMSYS). SBERT encoding requires GPU.

---

## Reproducibility

All experiments ran on a single machine with an NVIDIA A100 GPU (concept extraction and clustering on CPU; SBERT encoding on GPU).

| Package | Version |
|:--------|:-------:|
| Python | 3.10 |
| spaCy | 3.8.11 |
| en_core_web_sm | 3.8.0 |
| scikit-learn | 1.7.2 |
| sentence-transformers | 3.3.0 |
| PyTorch | 2.1.0+cu121 |
| igraph | 1.0.0 |
| leidenalg | 0.11.0 |

Leiden: 3 iterations, seed 42.

---

## Citation

```bibtex
@inproceedings{bhatia2026conceptknn,
  title={Concept-kNN: Interpretable Graph Clustering of LLM Conversations via Classical NLP},
  author={Bhatia, Divyansh and Mehala, N.},
  booktitle={Proc. IEEE CONECCT},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

We thank the creators of the [ShareGPT](https://sharegpt.com) and [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) datasets for making their data publicly available.
