"""FedNova strategy implementation for Flower.

Based on the paper: "Tackling the Objective Inconsistency Problem in Heterogeneous
Federated Optimization" (NeurIPS 2020)

FedNova normalizes client updates by their local step count to handle heterogeneity.
"""

from typing import List, Optional, Tuple, Union, Dict
from flwr.serverapp.strategy import FedAvg


class FedNova(FedAvg):
    """Federated Normalized Averaging (FedNova) strategy.

    FedNova addresses the objective inconsistency problem in heterogeneous
    federated optimization by normalizing local updates based on the number
    of local steps performed by each client.

    Key difference from FedAvg:
    - FedAvg aggregates: Δ_i = sum of gradients over τ_i steps
    - FedNova aggregates: Δ_i normalized by τ_i, then scaled by τ_eff

    This ensures fair contribution from clients with different local update counts.

    Parameters
    ----------
    var_local_epochs : bool, optional
        Whether to allow variable local epochs per client (default: False).
        If True, clients may perform different numbers of local updates.
        Currently, we use fixed local epochs for simplicity.
    **kwargs
        Additional arguments passed to FedAvg.
    """

    def __init__(
        self,
        var_local_epochs: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.var_local_epochs = var_local_epochs
        # Track tau_eff (effective number of local steps) across rounds
        self.tau_eff_history = []

    def aggregate_fit(
        self,
        server_round: int,
        replies,
        failures,
    ):
        """Aggregate fit results using FedNova normalized averaging.

        The key insight of FedNova:
        1. Extract tau_i from each client (number of local SGD steps)
        2. Calculate tau_eff = sum(tau_i * n_i) / sum(n_i)
        3. Normalize each client's weight by: a_i = (tau_eff / tau_i) * (n_i / sum(n_i))
        4. Aggregate: global_weights = sum(a_i * client_weights_i)

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        replies : List[Message]
            Successful replies from clients containing updated parameters.
        failures : List[Union[Message, BaseException]]
            Failed replies or exceptions from clients.

        Returns
        -------
        Tuple[Optional[Message], Dict[str, any]]
            Aggregated parameters and metrics.
        """
        if not replies:
            return None, {}

        # Extract tau_i, num_examples, and parameters from each client
        tau_values = []
        num_examples_list = []
        weights_results = []

        for reply in replies:
            # Extract parameters
            params = reply.content["arrays"]

            # Extract metrics including tau_i reported by client
            metrics = reply.content.get("metrics", {})
            num_examples = metrics.get("num-examples", 1)
            tau_i = metrics.get("tau", 1.0)  # Client reports number of local steps

            weights_results.append((params, num_examples))
            tau_values.append(tau_i)
            num_examples_list.append(num_examples)

        # Calculate tau_eff (effective number of local steps)
        # tau_eff = sum(tau_i * n_i) / sum(n_i)
        total_examples = sum(num_examples_list)
        if total_examples > 0:
            tau_eff = sum(tau_i * n_i for tau_i, n_i in zip(tau_values, num_examples_list)) / total_examples
        else:
            tau_eff = 1.0

        self.tau_eff_history.append(tau_eff)

        # FedNova normalized aggregation
        # For each client i: normalized_weight_i = (tau_eff / tau_i) * (n_i / total_examples)
        normalized_weights_results = []
        for (params, num_examples), tau_i in zip(weights_results, tau_values):
            # Calculate normalized weight for this client
            if tau_i > 0:
                normalized_weight = (tau_eff / tau_i) * (num_examples / total_examples)
            else:
                normalized_weight = num_examples / total_examples

            normalized_weights_results.append((params, normalized_weight * total_examples))

        # Use parent class aggregation with normalized weights
        # Create a modified replies list with normalized weights
        modified_replies = []
        for i, reply in enumerate(replies):
            # Create new metrics with normalized num_examples
            _, normalized_num = normalized_weights_results[i]
                # Create modified reply with normalized weights
            # We'll modify the metrics directly
            import copy
            modified_reply = copy.copy(reply)
            modified_content = copy.copy(reply.content)
            modified_metrics = copy.copy(reply.content.get("metrics", {}))
            modified_metrics["num-examples"] = int(normalized_num)
            modified_content["metrics"] = modified_metrics
            modified_reply.content = modified_content
            modified_replies.append(modified_reply)

        aggregated_result, metrics = super().aggregate_fit(server_round, modified_replies, failures)

        # Add FedNova-specific metrics
        metrics["tau_eff"] = tau_eff
        metrics["tau_mean"] = sum(tau_values) / len(tau_values) if tau_values else 0
        metrics["tau_min"] = min(tau_values) if tau_values else 0
        metrics["tau_max"] = max(tau_values) if tau_values else 0

        return aggregated_result, metrics

    def __repr__(self) -> str:
        """Return string representation of strategy."""
        return (
            f"FedNova("
            f"var_local_epochs={self.var_local_epochs}, "
            f"fraction_fit={self.fraction_fit}, "
            f"fraction_evaluate={self.fraction_evaluate})"
        )
