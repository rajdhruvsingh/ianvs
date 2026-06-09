# CIFAR-100 Federated Class-Incremental Semi-Supervised Learning

This directory benchmarks **Federated Class-Incremental Learning (FCIL)**
algorithms on CIFAR-100, extending standard federated learning to handle
new classes arriving over time across distributed clients.

## Overview

In real-world edge deployments, new categories of data appear continuously
while older data may no longer be accessible. This directory evaluates
algorithms that handle this challenge — learning new classes without
forgetting previously learned ones (catastrophic forgetting) in a
federated setting.

## Paradigm

`federatedclassincrementallearning` — combines federated learning with
class-incremental learning across distributed clients.

## Algorithms

Four FCIL algorithms are implemented and benchmarkable:

| Algorithm | Directory | Description |
|---|---|---|
| **FedAvg** | `fedavg/` | Baseline federated averaging adapted for FCIL |
| **GLFC** | `glfc/` | Global-Local Forgetting Compensation |
| **Fed-CI-Match** | `fed_ci_match/` | Federated class-incremental matching |
| **Fed-CI-Match-v2** | `fed_ci_match_v2/` | Improved Fed-CI-Match with better stability |

## Dataset

**CIFAR-100**: 60,000 images across 100 classes (600 per class).
- 50,000 training / 10,000 test images
- Backend: TensorFlow
- Classes split into incremental tasks

## Metrics

- **accuracy**: Overall classification accuracy
- **task_avg_acc**: Average accuracy across all incremental tasks
- **forget_rate**: Catastrophic forgetting rate across tasks

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
# update train_url and test_url in each algorithm's testenv/testenv.yaml
```

**Step 2 — Run FedAvg baseline:**
```bash
ianvs -f examples/cifar100/fci_ssl/fedavg/benchmarkingjob.yaml
```

**Step 3 — Run GLFC:**
```bash
ianvs -f examples/cifar100/fci_ssl/glfc/benchmarkingjob.yaml
```

**Step 4 — Compare results:**

Results are saved to the workspace directory. Compare `task_avg_acc`
and `forget_rate` across algorithms to evaluate FCIL performance.

## Key Differences From Standard FL

| Feature | Standard FL (`federated_learning/`) | FCIL (`fci_ssl/`) |
|---|---|---|
| Classes | Fixed across all rounds | New classes added incrementally |
| Forgetting | Not evaluated | Measured via `forget_rate` |
| Metrics | accuracy only | accuracy + task_avg_acc + forget_rate |
| Algorithms | FedAvg only | FedAvg, GLFC, Fed-CI-Match, Fed-CI-Match-v2 |

## Directory Structure

```
fci_ssl/
├── fedavg/                    # FedAvg baseline for FCIL
│   ├── algorithm/             # Model and aggregation code
│   ├── testenv/               # Metrics and evaluation
│   └── benchmarkingjob.yaml   # Run configuration
├── glfc/                      # GLFC algorithm
├── fed_ci_match/              # Fed-CI-Match algorithm
└── fed_ci_match_v2/           # Fed-CI-Match v2
```