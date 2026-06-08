# test_516_517.py
import yaml
import os
import sys

print("Testing Issue #516 — acc.py path fixes...")
examples = [
    "examples/cifar100/fci_ssl/fedavg/testenv/testenv.yaml",
    "examples/cifar100/fci_ssl/glfc/testenv/testenv.yaml",
    "examples/cifar100/fci_ssl/fed_ci_match_v2/testenv/testenv.yaml",
    "examples/cifar100/fci_ssl/fed_ci_match/testenv/testenv.yaml",
    "examples/cifar100/federated_class_incremental_learning/fedavg/testenv/testenv.yaml",
    "examples/cifar100/federated_learning/fedavg/testenv/testenv.yaml",
]

all_pass = True
for path in examples:
    with open(path) as f:
        content = f.read()
    has_hardcoded = "/home/wyd/" in content
    has_relative = "./examples/cifar100" in content
    status = "✅" if not has_hardcoded else "❌"
    print(f"  {status} {path.split('cifar100/')[1]}")
    if has_hardcoded:
        all_pass = False

print(f"\n{'✅ All acc.py paths fixed!' if all_pass else '❌ Some paths still hardcoded'}")

print("\nTesting Issue #517 — algorithms:conda typo fix...")
with open("examples/cifar100/fci_ssl/fedavg/benchmarkingjob.yaml") as f:
    bj_content = f.read()

has_typo = "algorithms:conda" in bj_content
try:
    yaml.safe_load(bj_content)
    yaml_valid = True
except yaml.YAMLError as e:
    yaml_valid = False
    print(f"  YAML error: {e}")

print(f"  {'✅' if not has_typo else '❌'} typo removed")
print(f"  {'✅' if yaml_valid else '❌'} YAML parses correctly")

if not all_pass or has_typo or not yaml_valid:
    sys.exit(1)