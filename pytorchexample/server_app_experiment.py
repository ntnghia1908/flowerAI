"""Experimental ServerApp with comprehensive logging for FL experiments."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from copy import deepcopy

from pytorchexample.task import Net, load_centralized_dataset
from pytorchexample.metrics import calculate_metrics, calculate_weight_metrics
from pytorchexample.logger import ExperimentLogger
from pytorchexample.strategies import get_strategy

# Create ServerApp
app = ServerApp()

# Global variables for tracking
previous_weights = None
experiment_logger = None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for experimental ServerApp with comprehensive logging."""
    global previous_weights, experiment_logger

    # Read run config
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)
    num_rounds: int = context.run_config.get("num-server-rounds", 500)
    lr: float = context.run_config.get("learning-rate", 0.1)
    num_clients: int = context.run_config.get("num-clients", 6)  # Fixed: 6 clients
    min_train_nodes: int = context.run_config.get("min-train-nodes", 6)  # All 6 clients train
    min_evaluate_nodes: int = context.run_config.get("min-evaluate-nodes", 6)  # All 6 clients evaluate

    # Experiment configuration
    strategy_name: str = context.run_config.get("strategy", "FedAvg")
    distribution: str = context.run_config.get("distribution", "homo")
    experiment_name: str = context.run_config.get("experiment-name", f"{strategy_name}_{distribution}")

    # Initialize experiment logger
    experiment_logger = ExperimentLogger(experiment_name)

    # Log experiment configuration
    config = {
        "strategy": strategy_name,
        "distribution": distribution,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": min_train_nodes,
        "min_evaluate_nodes": min_evaluate_nodes,
        "learning_rate": lr,
    }
    experiment_logger.log_experiment_config(config)

    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Strategy: {strategy_name}")
    print(f"Distribution: {distribution}")
    print(f"Rounds: {num_rounds}")
    print(f"Clients: {num_clients} (train: {min_train_nodes}, eval: {min_evaluate_nodes})")
    print(f"{'='*60}\n")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    previous_weights = deepcopy(global_model.state_dict())

    # Get strategy
    strategy = get_strategy(
        strategy_name=strategy_name,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_evaluate_nodes,
        min_available_nodes=num_clients,
    )

    # Start strategy, run FL for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"{experiment_name}_final_model.pt")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment completed: {experiment_name}")
    print(f"Results saved to:")
    summary = experiment_logger.get_summary()
    for key, path in summary.items():
        if key not in ['experiment_name', 'timestamp']:
            print(f"  - {key}: {path}")
    print(f"{'='*60}\n")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data with comprehensive metrics logging."""
    global previous_weights, experiment_logger

    # Load the model and initialize it with the received weights
    model = Net()
    current_weights = arrays.to_torch_state_dict()
    model.load_state_dict(current_weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Calculate comprehensive metrics
    metrics = calculate_metrics(model, test_dataloader, device)

    # Calculate weight metrics
    weight_metrics = calculate_weight_metrics(current_weights, previous_weights)

    # Update previous weights
    previous_weights = deepcopy(current_weights)

    # Log to CSV
    if experiment_logger is not None:
        experiment_logger.log_global_metrics(server_round, metrics)
        experiment_logger.log_weight_metrics(server_round, weight_metrics)

    # Print progress
    print(f"Round {server_round:3d} | "
          f"Loss: {metrics['loss']:.4f} | "
          f"Acc: {metrics['accuracy']:.4f} | "
          f"F1: {metrics['f1']:.4f} | "
          f"Weight Change: {weight_metrics['weight_relative_change']:.6f}")

    # Return metrics (Flower expects accuracy and loss keys)
    return MetricRecord({
        "accuracy": metrics['accuracy'],
        "loss": metrics['loss'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
        "weight_norm": weight_metrics['weight_norm'],
        "weight_change": weight_metrics['weight_change'],
        "weight_relative_change": weight_metrics['weight_relative_change'],
    })
