# Complete Guide - FL Experiments

## Quick Links
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Analysis](#analysis)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
pip install -e .
```

## Configuration

### 6 Clients Setup
```bash
cp configs/config_6clients.toml pyproject.toml
```

### 10 Clients Setup
```bash
cp configs/config_10clients.toml pyproject.toml
```

## Running Experiments

### Quick Test
```bash
python scripts/test_quick.py
```

### Run Experiments
```bash
# Quick (10 rounds)
python scripts/run_experiments.py --quick --yes

# Medium (100 rounds)
python scripts/run_experiments.py --test --yes

# Full (500 rounds)
python scripts/run_experiments.py --all --yes
```

## Analysis

```bash
python scripts/analyze_results.py
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
