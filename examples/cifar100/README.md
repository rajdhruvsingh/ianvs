# CIFAR-100 Federated Class-Incremental Learning Benchmarking

This example benchmarks **Federated Class-Incremental Learning (FCIL)** 
algorithms on the CIFAR-100 dataset using the Ianvs benchmarking framework.

## Overview

In real-world edge scenarios, data arrives incrementally across distributed 
clients while new classes are introduced over time. This example evaluates 
how well federated learning algorithms handle class-incremental learning 
without forgetting previously learned classes (catastrophic forgetting).

## Paradigm

`federatedclassincrementallearning` — combines federated learning across 
multiple clients with class-incremental learning over time.

## Dataset

**CIFAR-100**: 60,000 images across 100 classes (600 images per class).
- 50,000 training images
- 10,000 test images
- Backend: TensorFlow

Prepare the dataset:
```bash
mkdir -p data/cifar100
# Download CIFAR-100 and generate index files
python examples/cifar100/utils.py
```

## Algorithms

Four federated class-incremental learning algorithms are implemented:

| Algorithm | Directory | Description |
|---|---|---|
| **FedAvg** | `fci_ssl/fedavg/` | Baseline federated averaging |
| **GLFC** | `fci_ssl/glfc/` | Global-Local Forgetting Compensation |
| **Fed-CI-Match** | `fci_ssl/fed_ci_match/` | Federated class-incremental matching |
| **Fed-CI-Match-v2** | `fci_ssl/fed_ci_match_v2/` | Improved version of Fed-CI-Match |

There is also a standard federated learning baseline:

| Algorithm | Directory | Description |
|---|---|---|
| **FedAvg (FL)** | `federated_learning/fedavg/` | Standard federated averaging without incremental learning |
| **FedAvg (FCIL)** | `federated_class_incremental_learning/fedavg/` | FedAvg adapted for class-incremental setting |

## Metrics

- **accuracy**: Overall classification accuracy
- **task_avg_acc**: Average accuracy across all tasks
- **forget_rate**: Rate of forgetting previously learned classes

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
```

**Step 2 — Prepare initial model:**
```bash
mkdir -p init_model
# place cnn.pb under init_model/
```

**Step 3 — Run benchmarking (FedAvg example):**
```bash
ianvs -f examples/cifar100/fci_ssl/fedavg/benchmarkingjob.yaml
```

**Step 4 — Check results:**

Results are saved to the workspace directory defined in 
`benchmarkingjob.yaml`. A leaderboard is printed to console 
showing `task_avg_acc` and `forget_rate` for each algorithm.

## Configuration

Key hyperparameters in `algorithm/algorithm.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 64 | Training batch size per client |
| `learning_rate` | 0.001 | Learning rate |
| `epochs` | 1 | Local training epochs per round |
| `train_ratio` | 1.0 | Ratio of data used for training |
| `data_partition` | iid | Data distribution across clients (iid/non-iid) |
| `incremental_rounds` | 2 | Number of incremental learning rounds |
| `client_number` | 1 | Number of federated clients |

## Directory Structure
cifar100/
├── fci_ssl/                          # Federated Class-Incremental SSL
│   ├── fedavg/                       # FedAvg algorithm
│   ├── glfc/                         # GLFC algorithm
│   ├── fed_ci_match/                 # Fed-CI-Match algorithm
│   └── fed_ci_match_v2/              # Fed-CI-Match-v2 algorithm
├── federated_class_incremental_learning/
│   └── fedavg/                       # FedAvg for FCIL paradigm
├── federated_learning/
│   └── fedavg/                       # Standard FedAvg baseline
└── utils.py                          # Dataset utilities

## Related Issues

- Hardcoded absolute paths fixed in PR #519
- For questions or issues, see the 
  [Ianvs issue tracker](https://github.com/kubeedge/ianvs/issues)