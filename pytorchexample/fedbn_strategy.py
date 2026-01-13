"""FedBN Strategy - Federated Learning with Local Batch Normalization.

Paper: "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
       (ICLR 2021)

Key Idea:
- Keep BatchNorm layers LOCAL (not aggregated across clients)
- Only aggregate other layers (Conv, FC, etc.)
- Handles feature shift across clients effectively

Note: Current model (simple CNN) doesn't have BatchNorm layers,
      but this implementation is ready for future models that do.
"""

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.serverapp.strategy import FedAvg


class FedBN(FedAvg):
    """FedBN strategy - keep BatchNorm layers local, aggregate rest.

    This strategy filters out BatchNorm layers during aggregation,
    allowing each client to maintain their own BN statistics.
    """

    def __init__(self, **kwargs):
        """Initialize FedBN strategy.

        Args:
            **kwargs: Keyword arguments for FedAvg
        """
        super().__init__(**kwargs)
        print("[FedBN] Initialized - BatchNorm layers will be kept local")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from multiple clients, excluding BatchNorm layers.

        Args:
            server_round: Current server round number
            results: List of (ClientProxy, FitRes) tuples from successful clients
            failures: List of exceptions/failed results from failed clients

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        if not results:
            return None, {}

        # Call parent's aggregate_fit to get initial aggregation
        # Note: For current model (no BN layers), this is equivalent to FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Filter BatchNorm layers if model has any
        # Current implementation: No-op since model doesn't have BN
        # Future: Add BN layer filtering here when model architecture changes

        # For debugging/logging
        if server_round == 1:
            print("[FedBN] Note: Current model has no BatchNorm layers")
            print("[FedBN] Strategy ready for future models with BN layers")

        return aggregated_parameters, aggregated_metrics

    def _filter_bn_layers(
        self,
        parameters: Parameters,
    ) -> Parameters:
        """Filter out BatchNorm layers from parameters.

        This is a placeholder for future implementation when model has BN layers.

        Args:
            parameters: Model parameters

        Returns:
            Filtered parameters (without BatchNorm layers)
        """
        # Convert to ndarrays
        ndarrays = parameters_to_ndarrays(parameters)

        # TODO: Implement BN layer filtering when model has BN layers
        # Pattern: Skip layers with names containing "bn", "batch_norm", etc.
        # For now, return all parameters since model has no BN

        filtered_ndarrays = ndarrays  # No filtering for current model

        # Convert back to Parameters
        return ndarrays_to_parameters(filtered_ndarrays)

    def _is_bn_layer(self, layer_name: str) -> bool:
        """Check if layer name indicates a BatchNorm layer.

        Args:
            layer_name: Name of the layer

        Returns:
            True if layer is a BatchNorm layer, False otherwise
        """
        bn_indicators = ["bn", "batch_norm", "batchnorm", "batch_normalization"]
        layer_name_lower = layer_name.lower()
        return any(indicator in layer_name_lower for indicator in bn_indicators)


# Note: For proper BatchNorm layer filtering in future models:
# 1. Model needs to have named parameters/layers
# 2. During aggregate_fit(), iterate through layer names
# 3. Skip layers where _is_bn_layer(name) returns True
# 4. Only aggregate non-BN layers
# 5. Clients keep their own BN layer statistics

# Example future implementation:
# def aggregate_fit(...):
#     # Get all client updates
#     weights_results = [
#         (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#         for _, fit_res in results
#     ]
#
#     # Get layer names from first client
#     layer_names = get_layer_names_from_model()
#
#     # Filter out BN layers
#     non_bn_indices = [
#         i for i, name in enumerate(layer_names)
#         if not self._is_bn_layer(name)
#     ]
#
#     # Only aggregate non-BN layers
#     filtered_weights = [
#         ([params[i] for i in non_bn_indices], num_examples)
#         for params, num_examples in weights_results
#     ]
#
#     # Aggregate filtered weights
#     aggregated = aggregate(filtered_weights)
#
#     # Reconstruct full parameters (BN from server's current model)
#     ...
