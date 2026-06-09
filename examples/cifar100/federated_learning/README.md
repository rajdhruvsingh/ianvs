# CIFAR-100 Federated Learning Benchmarking

This directory contains a standard **Federated Learning (FL)** baseline
for CIFAR-100 image classification using the FedAvg algorithm.

## Overview

Federated Learning trains a shared model across multiple distributed
clients without centralizing raw data. Each client trains locally and
only shares model updates with the aggregation server, preserving
data privacy.

This example serves as the **standard FL baseline** for comparison
against the more advanced Federated Class-Incremental Learning (FCIL)
algorithms in `examples/cifar100/fci_ssl/`.

## Paradigm

`federatedlearning` — standard federated averaging across distributed
clients with IID data partitioning.

## Algorithm

| Algorithm | Directory | Description |
|---|---|---|
| **FedAvg** | `fedavg/` | Standard federated averaging baseline |

## Dataset

**CIFAR-100**: 60,000 images across 100 classes.
- Backend: TensorFlow
- Data partition: IID (evenly distributed across clients)

## Metrics

- **accuracy**: Overall classification accuracy across all classes

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 32 | Training batch size per client |
| `learning_rate` | 0.001 | Learning rate |
| `epochs` | 10 | Local training epochs per round |
| `train_ratio` | 1.0 | Ratio of data used for training |
| `data_partition` | iid | Data distribution (iid/non-iid) |
| `incremental_rounds` | 10 | Number of federated rounds |
| `round` | 200 | Total training rounds |

## Prerequisites

- Python >= 3.8
- TensorFlow
- Ianvs installed (`python setup.py install`)
- KubeEdge Sedna (`pip install resources/third_party/sedna-0.6.0.1-py3-none-any.whl`)

## Quick Start

**Step 1 — Prepare dataset:**
```bash
mkdir -p data/cifar100
# place cifar100_train.txt and cifar100_test.txt under data/cifar100/
# update train_url and test_url in testenv/testenv.yaml
```

**Step 2 — Prepare initial model:**
```bash
mkdir -p init_model
# place restnet.pb under init_model/
# update initial_model_url in algorithm/algorithm.yaml
```

**Step 3 — Run benchmarking:**
```bash
ianvs -f examples/cifar100/federated_learning/fedavg/benchmarkingjob.yaml
```

## Relationship to Other cifar100 Examples
cifar100/
├── federated_learning/          ← YOU ARE HERE (standard FL baseline)
├── fci_ssl/                     ← Federated Class-Incremental Learning
│   ├── fedavg/                  ← FedAvg with class-incremental setting
│   ├── glfc/                    ← GLFC algorithm
│   ├── fed_ci_match/            ← Fed-CI-Match algorithm
│   └── fed_ci_match_v2/         ← Fed-CI-Match-v2 algorithm
└── federated_class_incremental_learning/
└── fedavg/                  ← FedAvg for FCIL paradigm

Use this example as a baseline before running the more advanced
FCIL algorithms in `fci_ssl/`.