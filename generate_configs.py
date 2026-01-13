"""Generate all strategy × distribution config files."""

from pathlib import Path

STRATEGIES = {
    "FedAvg": {},
    "FedAvgM": {"server-momentum": 0.9, "server-learning-rate": 0.5},
    "FedProx": {"proximal-mu": 0.01},
    "FedAdam": {"eta": 0.01, "eta-l": 0.1, "beta-1": 0.9, "beta-2": 0.99, "tau": 1e-9},
    "FedAdagrad": {"eta": 0.01, "eta-l": 0.1, "tau": 1e-9},
    "FedYogi": {"eta": 0.01, "eta-l": 0.1, "beta-1": 0.9, "beta-2": 0.99, "tau": 1e-9},
    "FedNova": {"var-local-epochs": False},
    "SCAFFOLD": {},
    "FedBN": {},
}

DISTRIBUTIONS = ["homo", "C2", "C3", "C4", "C5", "Dir0.1", "Dir0.5", "Dir1.0", "Dir10.0"]

BASE_CONFIG = """# {strategy} on {distribution} distribution
num-server-rounds = 500
fraction-train = 1.0
fraction-evaluate = 1.0
min-train-nodes = 6
min-evaluate-nodes = 6
local-epochs = 1
learning-rate = 0.01
batch-size = 64
strategy = "{strategy}"
distribution = "{distribution}"
experiment-name = "{strategy}_{distribution}_npy"
num-clients = 6
data-source = "npy"

# Strategy-specific parameters (declared in pyproject.toml)
# All parameters must be declared in pyproject.toml even if not used by this strategy
server-momentum = 0.9
server-learning-rate = 0.5
proximal-mu = 0.01
eta = 0.01
eta-l = 0.1
tau = 1e-9
beta-1 = 0.9
beta-2 = 0.99
var-local-epochs = false
"""

def generate_config(strategy, distribution, params):
    """Generate a single config file."""
    # Don't add extra_params to avoid duplication - all params are already in BASE_CONFIG
    content = BASE_CONFIG.format(
        strategy=strategy,
        distribution=distribution
    )

    filename = f"configs/{strategy}_{distribution}_npy.toml"
    Path(filename).parent.mkdir(exist_ok=True)

    with open(filename, 'w') as f:
        f.write(content)

    print(f"Generated: {filename}")

def main():
    """Generate all configs."""
    total = 0
    for strategy, params in STRATEGIES.items():
        for distribution in DISTRIBUTIONS:
            generate_config(strategy, distribution, params)
            total += 1

    print(f"\nTotal generated: {total} config files")

if __name__ == '__main__':
    main()
