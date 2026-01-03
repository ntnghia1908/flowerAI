"""Server app using pre-partitioned .npy data for centralized evaluation."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

import torch

from pytorchexample.task_experiment import Net, get_weights, set_weights
from pytorchexample.task_npy import load_npy_centralized_test, get_data_dir
from pytorchexample.metrics import calculate_metrics
from pytorchexample.logger import ExperimentLogger

# Global variables for tracking
current_round = 0
experiment_logger = None
previous_weights = None
client_aggregate_metrics = {'global_accuracy': 0.0, 'weighted_accuracy': 0.0}


class CustomFedAvg(FedAvg):
    """Custom FedAvg with client metrics aggregation."""

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results."""
        aggregated_result = super().aggregate_fit(server_round, results, failures)

        if aggregated_result is None:
            return None

        # Log client training metrics
        global experiment_logger
        if experiment_logger:
            for client_proxy, fit_res in results:
                client_metrics = fit_res.metrics
                experiment_logger.log_client_metrics(
                    server_round,
                    int(client_proxy.cid),
                    client_metrics,
                    is_training=True
                )

        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and calculate global/weighted accuracy."""
        global current_round, experiment_logger, client_aggregate_metrics

        aggregated_result = super().aggregate_evaluate(server_round, results, failures)

        # Initialize lists for collecting client metrics
        client_accuracies = []
        client_num_examples = []

        # Log client evaluation metrics and collect data
        if experiment_logger:
            for client_proxy, evaluate_res in results:
                metrics = evaluate_res.metrics
                experiment_logger.log_client_metrics(
                    server_round,
                    int(client_proxy.cid),
                    metrics,
                    is_training=False
                )

                # Collect client accuracies and sample counts
                client_accuracies.append(metrics.get('eval_acc', 0.0))
                client_num_examples.append(metrics.get('num-examples', 0))

        # Calculate aggregate metrics
        N = len(client_accuracies)
        if N > 0:
            # Global Accuracy = (1/N) × Σ(Accuracy_k)
            global_accuracy = sum(client_accuracies) / N

            # Weighted Accuracy = Σ(n_k × Accuracy_k) / Σ(n_k)
            total_examples = sum(client_num_examples)
            if total_examples > 0:
                weighted_accuracy = sum(
                    acc * n for acc, n in zip(client_accuracies, client_num_examples)
                ) / total_examples
            else:
                weighted_accuracy = 0.0
        else:
            global_accuracy = 0.0
            weighted_accuracy = 0.0

        # Store in global variable for use in global_evaluate
        client_aggregate_metrics = {
            'global_accuracy': global_accuracy,
            'weighted_accuracy': weighted_accuracy
        }

        return aggregated_result


def global_evaluate(server_round, parameters, config):
    """Evaluate model on centralized test set using .npy data."""
    global current_round, experiment_logger, previous_weights, client_aggregate_metrics

    # Get run config
    distribution = config.get("distribution", "homo")
    num_clients = config.get("num-clients", 6)
    data_base_dir = config.get("data-dir", "./data")

    # Create model and set weights
    model = Net()
    current_weights = parameters.tensors
    set_weights(model, current_weights)

    # Move to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get data directory and load centralized test set
    data_dir = get_data_dir(distribution, num_clients, data_base_dir)
    test_dataloader = load_npy_centralized_test(data_dir)

    # Calculate comprehensive metrics
    metrics = calculate_metrics(model, test_dataloader, device)

    # Calculate weight metrics
    from pytorchexample.metrics import calculate_weight_metrics
    weight_metrics = calculate_weight_metrics(current_weights, previous_weights)

    # Merge all metrics (centralized + client aggregates + weight)
    combined_metrics = {
        **metrics,  # centralized test metrics
        **client_aggregate_metrics,  # client aggregate metrics
        **weight_metrics  # weight change metrics
    }

    # Log to CSV
    if experiment_logger:
        experiment_logger.log_global_metrics(server_round, combined_metrics)
        experiment_logger.log_weight_metrics(server_round, weight_metrics)

    # Print progress
    print(
        f"Round {server_round:3d} | "
        f"Loss: {metrics['loss']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"Global Acc: {client_aggregate_metrics['global_accuracy']:.4f} | "
        f"Weighted Acc: {client_aggregate_metrics['weighted_accuracy']:.4f} | "
        f"Weight Change: {weight_metrics['weight_change']:.6f}"
    )

    # Update previous weights
    previous_weights = [w.copy() for w in current_weights]

    # Return results
    return metrics['loss'], {
        **combined_metrics
    }


def main(grid, context: Context):
    """Main server function using pre-partitioned .npy data."""
    global current_round, experiment_logger, previous_weights

    # Get config
    num_rounds = context.run_config["num-server-rounds"]
    distribution = context.run_config.get("distribution", "homo")
    experiment_name = context.run_config.get("experiment-name", "experiment")

    # Initialize logger
    experiment_logger = ExperimentLogger(experiment_name)

    # Print experiment info
    print("\n" + "=" * 60)
    print(f"Starting Experiment: {experiment_name}")
    print(f"Strategy: FedAvg")
    print(f"Distribution: {distribution}")
    print(f"Rounds: {num_rounds}")
    print(f"Data source: Pre-partitioned .npy files")
    print("=" * 60)
    print()

    # Initialize model and get initial parameters
    net = Net()
    parameters = ndarrays_to_parameters(get_weights(net))
    previous_weights = get_weights(net)

    # Create strategy
    strategy = CustomFedAvg(
        fraction_fit=context.run_config.get("fraction-train", 1.0),
        fraction_evaluate=context.run_config.get("fraction-evaluate", 1.0),
        min_fit_clients=context.run_config.get("min-train-nodes", 6),
        min_evaluate_clients=context.run_config.get("min-evaluate-nodes", 6),
        min_available_clients=context.run_config.get("min-evaluate-nodes", 6),
        initial_parameters=parameters,
        evaluate_fn=lambda round, params, config: global_evaluate(
            round, params, context.run_config
        ),
    )

    # Run strategy
    config = ServerConfig(num_rounds=num_rounds)
    result = strategy.start(grid=grid, config=config, context=context)

    # Save final model
    final_model_path = f"{experiment_name}_final_model.pt"
    torch.save(net.state_dict(), final_model_path)

    print("\n" + "=" * 60)
    print(f"Experiment completed: {experiment_name}")
    print(f"Results saved to:")
    print(f"  - global_csv: {experiment_logger.global_csv_path}")
    print(f"  - client_csv: {experiment_logger.client_csv_path}")
    print(f"  - weight_csv: {experiment_logger.weight_csv_path}")
    print("=" * 60)
    print()

    return result


# Create ServerApp
app = ServerApp(main=main)
