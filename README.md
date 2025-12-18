# FORGETTING: A New Mechanism Towards Better Large Language Model Fine-Tuning

<a href='https://github.com/AliTaheri2002/Forgetting-A-New-Mechanism-Towards-Better-Large-Language-Model-Fine-tuning'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2508.04329'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>

[Ali Taheri](https://alitaheri2002.github.io/), [Alireza Taban](https://www.linkedin.com/in/alireza-taban-90a460121/), [Qizhou Wang](https://qizhouwang.github.io/homepage/), [Shanshan Ye](https://cassie133ye.github.io/), [Abdolreza Mirzaei](https://people.iut.ac.ir/en/mirzaei), [Tongliang Liu](https://tongliang-liu.github.io/), [Bo Han](https://bhanml.github.io/).

Isfahan University of Technology, Hong Kong Baptist University, University of Technology Sydney, Simon Fraser University, The University of Sydney

-----
### Brief Introduction

This project investigates a novel forgetting mechanism for supervised fine-tuning of large language models. Our method categorizes tokens into positive and negative sets based on influence scores, then learns from positive tokens while actively forgetting negative tokens. This approach maintains full dataset scale while establishing clearer knowledge boundaries, leading to improved generalization and performance across multiple benchmarks.

## Dataset

The dataset used in this work is available at: https://huggingface.co/datasets/Aligh1380/forgetting-llm-dataset

```python
from datasets import load_dataset
dataset = load_dataset("Aligh1380/forgetting-llm-dataset")
```

## Overview

Traditional supervised fine-tuning approaches either use all data uniformly (potentially learning from noise) or discard low-quality samples (reducing dataset scale). Our FORGETTING mechanism addresses this by:

1. **Reference Model Fine-tuning**: Train a reference model on a subset of data
2. **Token Quality Assessment**: Calculate influence scores for each token
3. **Token Selection**: Partition tokens into positive and negative sets
4. **Training with Forgetting**: Train using dual objective with adaptive coefficient

## Usage

### Model Selection
Edit `run_pipeline.sh` to select your base model:

```bash
# Base models - uncomment one:
# base_model="meta-llama/Llama-3.2-1B"    # 1B model
# base_model="meta-llama/Llama-3.2-3B"    # 3B model
base_model="meta-llama/Llama-3.1-8B"      # 8B model
```

### Training
Run the complete pipeline:

```bash
bash run_pipeline.sh
```

### Evaluation
Evaluate the trained model:

```bash
bash run_evaluation.sh <model_size>
```

## Results

Performance improvements over standard SFT:

| Model | Standard SFT | Forgetting (Ours) | Improvement |
|-------|--------------|-------------------|-------------|
| LLaMA-3.2-1B | 30.37% | 34.86% | +4.49% |
| LLaMA-3.2-3B | 46.90% | 52.18% | +5.28% |
| LLaMA-3.1-8B | 51.02% | 59.27% | +8.25% |

## Citation

```bibtex
@article{ghahrizjani2025forgetting,
  title={Forgetting: A New Mechanism Towards Better Large Language Model Fine-Tuning},
  author={Ghahrizjani, Ali Taheri and Taban, Alireza and Wang, Qizhou and Ye, Shanshan and Mirzaei, Abdolreza and Liu, Tongliang and Han, Bo},
  journal={arXiv preprint arXiv:2508.04329},
  year={2025}
}
```
