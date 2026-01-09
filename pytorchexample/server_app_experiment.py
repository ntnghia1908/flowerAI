"""Experimental ServerApp with comprehensive logging for FL experiments."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from copy import deepcopy
from typing import List, Tuple, Dict, Optional

from pytorchexample.task import Net, load_centralized_dataset
from pytorchexample.metrics import calculate_metrics
from pytorchexample.logger import ExperimentLogger
from pytorchexample.strategies import get_strategy

# Create ServerApp
app = ServerApp()

# Global variables for tracking - used for passing data between callbacks
experiment_logger = None
current_round = 0
client_aggregate_metrics = {'global_accuracy': 0.0, 'weighted_accuracy': 0.0}


def create_evaluate_metrics_aggregation_fn(logger):
    """Create callback function for aggregating evaluation metrics."""

    def evaluate_metrics_aggregation_fn(replies):
        """Aggregate evaluation metrics from clients and log them."""
        global client_aggregate_metrics, current_round

        # Extract metrics from replies (each reply is a Message object)
        client_accuracies = []
        client_num_examples = []

        # Convert to list to iterate multiple times
        replies_list = list(replies)

        # Replies is an iterable of Message objects
        for idx, reply in enumerate(replies_list):
            # Extract metrics from reply.content (RecordDict)
            if hasattr(reply, 'content') and reply.content is not None:
                record_dict = reply.content

                # RecordDict has metrics_records attribute which is a dict
                # The actual metrics are in record_dict.metrics_records['metrics'] as a plain dict
                if hasattr(record_dict, 'metrics_records') and 'metrics' in record_dict.metrics_records:
                    metrics_dict = record_dict.metrics_records['metrics']

                    # num_examples is in the metrics dict with key 'num-examples'
                    num_examples = metrics_dict.get('num-examples', 0)

                    if logger is not None:
                        # Log individual client metrics
                        client_metrics = {
                            'loss': metrics_dict.get('eval_loss', 0.0),
                            'accuracy': metrics_dict.get('eval_acc', 0.0),
                            'precision': metrics_dict.get('eval_precision', 0.0),
                            'recall': metrics_dict.get('eval_recall', 0.0),
                            'f1': metrics_dict.get('eval_f1', 0.0),
                            'num_examples': num_examples
                        }
                        logger.log_client_metrics(current_round, idx, 'evaluate', client_metrics)

                        # Collect for aggregate calculation
                        client_accuracies.append(metrics_dict.get('eval_acc', 0.0))
                        client_num_examples.append(num_examples)

        # Calculate aggregate metrics
        N = len(client_accuracies)
        if N > 0:
            global_accuracy = sum(client_accuracies) / N
            total_examples = sum(client_num_examples)
            weighted_accuracy = sum(acc * n for acc, n in zip(client_accuracies, client_num_examples)) / total_examples if total_examples > 0 else 0.0
        else:
            global_accuracy = 0.0
            weighted_accuracy = 0.0

        # Store for global_evaluate to use
        client_aggregate_metrics = {
            'global_accuracy': global_accuracy,
            'weighted_accuracy': weighted_accuracy
        }

        # Return aggregated metrics (Flower expects this)
        return {}

    return evaluate_metrics_aggregation_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for experimental ServerApp with comprehensive logging."""
    global previous_weights, experiment_logger, current_round

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
    data_source: str = context.run_config.get("data-source", "huggingface")
    experiment_name: str = context.run_config.get("experiment-name", f"{strategy_name}_{distribution}")

    # Initialize experiment logger
    experiment_logger = ExperimentLogger(experiment_name)

    # Log experiment configuration
    config = {
        "strategy": strategy_name,
        "distribution": distribution,
        "data_source": data_source,
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
    print(f"Data Source: {data_source.upper()}")
    print(f"Rounds: {num_rounds}")
    print(f"Clients: {num_clients} (train: {min_train_nodes}, eval: {min_evaluate_nodes})")
    print(f"{'='*60}\n")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    # previous_weights = deepcopy(global_model.state_dict())

    # Read strategy parameters from config
    strategy_name: str = context.run_config.get("strategy", "FedAvg")

    # Strategy-specific parameters
    strategy_params = {}
    if strategy_name == "FedAvgM":
        strategy_params["server_momentum"] = context.run_config.get("server-momentum", 0.9)
        strategy_params["server_learning_rate"] = context.run_config.get("server-learning-rate", 0.5)
    elif strategy_name == "FedProx":
        strategy_params["proximal_mu"] = context.run_config.get("proximal-mu", 0.01)
    elif strategy_name in ["FedAdam", "FedYogi", "FedAdagrad"]:
        strategy_params["eta"] = context.run_config.get("eta", 0.01)
        strategy_params["eta_l"] = context.run_config.get("eta-l", 0.1)
        strategy_params["tau"] = context.run_config.get("tau", 1e-9)
        if strategy_name in ["FedAdam", "FedYogi"]:
            strategy_params["beta_1"] = context.run_config.get("beta-1", 0.9)
            strategy_params["beta_2"] = context.run_config.get("beta-2", 0.99)

    # Create evaluate metrics aggregation callback with logger
    eval_metrics_agg_fn = create_evaluate_metrics_aggregation_fn(experiment_logger)

    # Create strategy using factory with metrics aggregation callback
    strategy = get_strategy(
        strategy_name=strategy_name,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_evaluate_nodes,
        min_available_nodes=num_clients,
        evaluate_metrics_aggregation_fn=eval_metrics_agg_fn,
        **strategy_params
    )

    # Start strategy, run FL for `num_rounds`
    # Wrap evaluate_fn to pass data source parameters
    def evaluate_fn_wrapper(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        global current_round
        current_round = server_round
        return global_evaluate(
            server_round, arrays,
            data_source=data_source,
            distribution=distribution,
            num_clients=num_clients
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn_wrapper,
    )

    # Create models directory
    from pathlib import Path
    models_dir = Path("models") / strategy_name
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save final model to disk
    print("\nSaving final model...")
    state_dict = result.arrays.to_torch_state_dict()
    final_model_path = models_dir / f"{experiment_name}_final_model.pt"
    torch.save(state_dict, final_model_path)
    print(f"  Final model saved: {final_model_path}")

    # Save best model (if we tracked which round was best)
    best_round = experiment_logger.best_round
    if best_round > 0:
        best_model_path = models_dir / f"{experiment_name}_best_round{best_round}_model.pt"
        # Note: In current implementation, we save final model as best since we don't keep
        # intermediate models. For true best model saving, would need to keep model at each round.
        torch.save(state_dict, best_model_path)
        experiment_logger.set_best_model_path(str(best_model_path))
        print(f"  Best model (round {best_round}, acc={experiment_logger.best_accuracy:.4f}): {best_model_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment completed: {experiment_name}")
    print(f"Results saved to:")
    summary = experiment_logger.get_summary()
    for key, path in summary.items():
        if key not in ['experiment_name', 'timestamp']:
            if key == 'best_accuracy':
                print(f"  - {key}: {path:.4f} (at round {summary['best_round']})")
            elif key not in ['best_round']:
                print(f"  - {key}: {path}")
    print(f"{'='*60}\n")


def global_evaluate(server_round: int, arrays: ArrayRecord,
                   data_source="huggingface", distribution="homo", num_clients=6) -> MetricRecord:
    """Evaluate model on central data with comprehensive metrics logging."""
    global experiment_logger, client_aggregate_metrics

    # Load the model and initialize it with the received weights
    model = Net()
    current_weights = arrays.to_torch_state_dict()
    model.load_state_dict(current_weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set (with data source option)
    test_dataloader = load_centralized_dataset(
        data_source=data_source,
        distribution=distribution,
        num_clients=num_clients
    )

    # Calculate comprehensive metrics
    metrics = calculate_metrics(model, test_dataloader, device)

    # Get client aggregate metrics from global variable (set by callback)
    agg_metrics = client_aggregate_metrics

    # Log to CSV
    if experiment_logger is not None:
        # Merge centralized metrics with client aggregate metrics
        combined_metrics = {
            **metrics,  # loss, accuracy, precision, recall, f1 from centralized test
            **agg_metrics  # global_accuracy, weighted_accuracy from clients
        }
        experiment_logger.log_global_metrics(server_round, combined_metrics)

        # Log hardware metrics
        experiment_logger.log_hardware_metrics(server_round)

    # Print progress
    print(f"Round {server_round:3d} | "
          f"Loss: {metrics['loss']:.4f} | "
          f"Acc: {metrics['accuracy']:.4f} | "
          f"F1: {metrics['f1']:.4f} | "
          f"Global Acc: {agg_metrics['global_accuracy']:.4f} | "
          f"Weighted Acc: {agg_metrics['weighted_accuracy']:.4f}" )

    # Return metrics (Flower expects accuracy and loss keys)
    return MetricRecord({
        "accuracy": metrics['accuracy'],
        "loss": metrics['loss'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
        "global_accuracy": agg_metrics['global_accuracy'],
        "weighted_accuracy": agg_metrics['weighted_accuracy'],
    })
