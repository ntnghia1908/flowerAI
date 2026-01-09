"""Custom federated learning strategies with metrics aggregation support."""

from typing import Callable, Optional
from flwr.serverapp.strategy import (
    FedAvg, FedAvgM, FedProx, FedAdam, FedAdagrad, FedYogi,
)


class FedAvgWithMetricsAggregation(FedAvg):
    """FedAvg with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        """Aggregate evaluation metrics and call the aggregation callback."""
        # Call the callback if provided
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)

        # Call parent's aggregate_evaluate
        return super().aggregate_evaluate(server_round, replies)


class FedAvgMWithMetricsAggregation(FedAvgM):
    """FedAvgM with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)


class FedProxWithMetricsAggregation(FedProx):
    """FedProx with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)


class FedAdamWithMetricsAggregation(FedAdam):
    """FedAdam with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)


class FedAdagradWithMetricsAggregation(FedAdagrad):
    """FedAdagrad with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)


class FedYogiWithMetricsAggregation(FedYogi):
    """FedYogi with support for evaluate_metrics_aggregation_fn callback."""

    def __init__(
        self,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def aggregate_evaluate(self, server_round: int, replies):
        if self.evaluate_metrics_aggregation_fn is not None:
            self.evaluate_metrics_aggregation_fn(replies)
        return super().aggregate_evaluate(server_round, replies)
