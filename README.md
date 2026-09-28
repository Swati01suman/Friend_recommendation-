# Friend Recommendation with Graph Neural Networks

A PyTorch-based graph learning project for predicting friendships and recommending likely connections in social networks. The repository implements several GNN architectures for link prediction, including GCN, GAT, and LightGCN, and compares them against classical graph heuristics such as Common Neighbors, Adamic–Adar, Jaccard, Resource Allocation, and Preferential Attachment.

## Overview

This project models a social graph as a network of users and edges representing friendships. The goal is to learn latent node representations and score candidate links to estimate which new connections are most likely to form.

The system is built around:

- PyTorch and PyTorch Geometric
- Link prediction on social network graphs
- Automatic feature engineering for graph attributes
- Config-driven training via YAML
- Multi-seed benchmarking and evaluation
- Baseline comparison with ranking metrics

## Features

- GNN encoder architectures:
  - GCN
  - GAT
  - GIN / GNN variants
  - LightGCN
- Link prediction predictors:
  - dot product
  - cosine similarity
  - bilinear models
  - MLP-based score functions
- Structural feature generation:
  - degree-based features
  - PageRank
  - random-walk positional encodings
  - Laplacian positional encodings
- Automatic dataset loading and train/validation/test splits
- Early stopping and learning-rate scheduling
- Ranking metrics for recommendations:
  - Precision@K
  - Recall@K
  - NDCG@K
  - MRR
  - AUC / AP
- Benchmark pipeline for multiple seeds and models

## Project Structure

```text
.
├── configs/
│   └── default.yaml               # Default training/evaluation configuration
├── data/                          # Dataset files (e.g., SNAP ego-networks)
├── results/                       # Benchmark and evaluation outputs
├── src/
│   ├── data/                      # Dataset loaders and transforms
│   ├── evaluation/                # Metrics and baseline heuristics
│   ├── models/                    # GNN encoders and predictors
│   ├── training/                  # Training logic, losses, schedulers
│   └── utils/                     # Config loader, logging, helpers
├── ablation_study_layers.py       # Layer-depth ablation experiments
├── benchmark.py                   # Multi-seed benchmark runner
├── friend_recommendation_gat.py   # Standalone GAT-based model experiment
├── friend_recommendation_gcn.py   # Standalone GCN-based model experiment
├── friend_recommendation_gnn.py   # General GNN experiment entry point
├── train.py                       # Main training script
├── requirements.txt               # Project dependencies
├── requirements-dev.txt           # Development dependencies
├── CLAUDE.md                     # Project instructions for Claude Code
├── PROGRESS.md                   # Project roadmap and implementation status
├── learning_curves.png            # Example training curves
├── tsne_clusters.png             # Example t-SNE cluster visualization
└── README.md                     # Project documentation
Dataset
The project is designed for social graph datasets, with the primary workflow centered around SNAP Facebook ego-network data. The default configuration points to the local data/ directory.

Supported dataset categories in the configuration include:

facebook
twitter
gplus
ogbl-collab
ogbl-ppa
ogbl-ddi
Installation
Create a virtual environment and install the dependencies:

bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
For development tooling:

bash
pip install -r requirements-dev.txt
Quick Start
Train a model
bash
python train.py --config configs/default.yaml --model gcn
You can also override training settings directly:

bash
python train.py --model gat --epochs 200 --lr 0.01 --dataset facebook
Run benchmark comparison
bash
python benchmark.py
This will evaluate classical heuristics and neural models across multiple seeds and write a summary report to:

Text
results/benchmark_summary.md
Dry run configuration
bash
python train.py --config configs/default.yaml --dry-run
Configuration
The default hyperparameters live in:

YAML
configs/default.yaml
This file includes settings for:

random seed
device selection
dataset split ratios
feature construction
model architecture
predictor type
optimizer and loss settings
early stopping
LR scheduling
logging and checkpointing
Example configuration highlights:

YAML
seed: 42
model:
  type: "gcn"
  hidden_channels: 128
  out_channels: 64
  num_layers: 2
training:
  epochs: 100
  lr: 0.005
Model Workflow
The training pipeline follows this sequence:

Load graph dataset
Build train/validation/test link splits
Apply feature engineering
Initialize GNN encoder and link predictor
Train with binary cross entropy or BPR-style objectives
Evaluate on validation and test sets
Save the best checkpoint
Run baseline comparisons if requested
Evaluation Metrics
The project evaluates recommendations using a combination of:

AUC
Average Precision (AP)
Recall@K
NDCG@K
MRR
Precision@K
This allows the model to be compared both as a binary link predictor and as a ranking system for recommendation quality.

Outputs
The repo includes generated artifacts such as:

learning_curves.png
learning_curves_gat.png
learning_curves_gnn.png
tsne_clusters.png
tsne_clusters_gat.png
tsne_clusters_gnn.png
ablation_results.png
These visualizations help assess training stability, representation quality, and clustering behavior.

Notes
This repository is designed as a research-oriented and experimentation-friendly graph recommendation framework. It is useful for:

learning graph neural networks for recommendation
evaluating social link prediction models
comparing heuristic vs learned approaches
benchmarking GCN/GAT-style architectures on graph datasets
License
This repository does not currently include a license file in the root directory. If you plan to reuse or redistribute the code, verify repository ownership and licensing terms before publication or commercial use.

Contributing
Contributions are welcome. To contribute:

Fork the repository
Create a feature branch
Make your changes
Run the relevant training/benchmark command
Submit a pull request with a clear summary


Acknowledgements
This project uses graph learning and link prediction concepts from the PyTorch Geometric ecosystem and social network recommendation research. It is intended as a practical implementation and benchmarking setup for friend recommendation tasks.

