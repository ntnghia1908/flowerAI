"""Federated Learning strategies for experiments."""

from pytorchexample.custom_strategies import (
    FedAvgWithMetricsAggregation,
    FedAvgMWithMetricsAggregation,
    FedProxWithMetricsAggregation,
    FedAdamWithMetricsAggregation,
    FedAdagradWithMetricsAggregation,
    FedYogiWithMetricsAggregation,
    FedNovaWithMetricsAggregation,
    SCAFFOLDWithMetricsAggregation,
)


def get_strategy(
    strategy_name: str,
    fraction_train: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_train_nodes: int = 2,
    min_evaluate_nodes: int = 2,
    min_available_nodes: int = 2,
    **kwargs
):
    """Get federated learning strategy by name.

    Args:
        strategy_name: Name of the strategy
            - "FedAvg": Federated Averaging
            - "FedAvgM": FedAvg with server momentum
            - "FedProx": FedAvg with proximal term
            - "FedAdam": FedAvg with Adam optimizer
            - "FedAdagrad": FedAvg with Adagrad optimizer
            - "FedYogi": FedAvg with Yogi optimizer
        fraction_train: Fraction of clients used for training
        fraction_evaluate: Fraction of clients used for evaluation
        min_train_nodes: Minimum number of clients for training
        min_evaluate_nodes: Minimum number of clients for evaluation
        min_available_nodes: Minimum number of available clients
        **kwargs: Additional strategy-specific parameters including evaluate_metrics_aggregation_fn

    Returns:
        Strategy object from flwr.serverapp.strategy
    """
    # Extract the callback function if provided
    evaluate_metrics_agg_fn = kwargs.pop("evaluate_metrics_aggregation_fn", None)

    common_params = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": min_train_nodes,
        "min_evaluate_nodes": min_evaluate_nodes,
        "min_available_nodes": min_available_nodes,
    }

    if strategy_name == "FedAvg":
        return FedAvgWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params
        )

    elif strategy_name == "FedAvgM":
        # FedAvg with server-side momentum
        server_momentum = kwargs.get("server_momentum", 0.9)
        server_learning_rate = kwargs.get("server_learning_rate", 0.5)
        return FedAvgMWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params,
            server_momentum=server_momentum,
            server_learning_rate=server_learning_rate
        )

    elif strategy_name == "FedProx":
        # FedAvg with proximal term to keep local models close to global
        proximal_mu = kwargs.get("proximal_mu", 0.01)
        return FedProxWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params,
            proximal_mu=proximal_mu
        )

    elif strategy_name == "FedAdam":
        # FedAvg with adaptive learning rate (Adam optimizer on server)
        eta = kwargs.get("eta", 1e-2)
        eta_l = kwargs.get("eta_l", 1e-1)
        beta_1 = kwargs.get("beta_1", 0.9)
        beta_2 = kwargs.get("beta_2", 0.99)
        tau = kwargs.get("tau", 1e-9)
        return FedAdamWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau
        )

    elif strategy_name == "FedAdagrad":
        # FedAvg with Adagrad optimizer on server
        eta = kwargs.get("eta", 1e-2)
        eta_l = kwargs.get("eta_l", 1e-1)
        tau = kwargs.get("tau", 1e-9)
        return FedAdagradWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params,
            eta=eta,
            eta_l=eta_l,
            tau=tau
        )

    elif strategy_name == "FedYogi":
        # FedAvg with Yogi optimizer on server
        eta = kwargs.get("eta", 1e-2)
        eta_l = kwargs.get("eta_l", 1e-1)
        beta_1 = kwargs.get("beta_1", 0.9)
        beta_2 = kwargs.get("beta_2", 0.99)
        tau = kwargs.get("tau", 1e-9)
        return FedYogiWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau
        )

    elif strategy_name == "FedNova":
        # Federated Normalized Averaging
        var_local_epochs = kwargs.get("var_local_epochs", False)
        return FedNovaWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params,
            var_local_epochs=var_local_epochs
        )

    elif strategy_name == "SCAFFOLD":
        # Stochastic Controlled Averaging for Federated Learning
        return SCAFFOLDWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params
        )

    # For strategies not yet implemented in Flower, fall back to FedAvg
    elif strategy_name in ["FiOFL", "FLOCO"]:
        print(f"Warning: {strategy_name} not implemented, using FedAvg instead")
        return FedAvgWithMetricsAggregation(
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
            **common_params
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def get_available_strategies():
    """Get list of available strategy names."""
    return [
        "FedAvg",
        "FedAvgM",
        "FedProx",
        "FedAdam",
        "FedAdagrad",
        "FedYogi",
        "FedNova",
        "SCAFFOLD",
        # Not yet implemented
        # "FiOFL",
        # "FLOCO",
    ]
