# CIFAR-100 Sedna Federated Learning

This directory implements federated learning for CIFAR-100 using
**KubeEdge Sedna's** native federated learning framework, as opposed
to the custom FL implementation in `examples/cifar100/federated_learning/`.

## Overview

While `federated_learning/fedavg/` implements FL logic directly,
this example uses Sedna's built-in worker architecture with separate
train workers and aggregation workers — closer to a real distributed
deployment.

## Architecture
sedna_federated_learning/
├── train_worker/              # Client-side training
│   ├── basemodel.py           # Model definition and local training
│   └── train.py               # Training entry point
└── aggregation_worker/        # Server-side aggregation
└── aggregate.py           # FedAvg aggregation logic

## Key Difference From `federated_learning/`

| Aspect | `federated_learning/` | `sedna_federated_learning/` |
|---|---|---|
| Framework | Custom Ianvs FL | Sedna native FL workers |
| Deployment | Single machine simulation | Distributed worker nodes |
| Entry point | `ianvs -f benchmarkingjob.yaml` | Separate train/aggregate workers |

## Prerequisites

- Python >= 3.8
- TensorFlow
- KubeEdge Sedna installed
- Multiple worker nodes (or simulated locally)

## Usage

**Start aggregation worker:**
```bash
python examples/cifar100/sedna_federated_learning/aggregation_worker/aggregate.py
```

**Start train workers (one per client):**
```bash
python examples/cifar100/sedna_federated_learning/train_worker/train.py
```