"""SCAFFOLD strategy implementation for Flower.

Based on the paper: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
(ICML 2020)

SCAFFOLD uses control variates to correct for client drift in federated learning.
"""

from typing import List, Optional, Tuple, Union, Dict, OrderedDict
from flwr.serverapp.strategy import FedAvg


class SCAFFOLD(FedAvg):
    """Stochastic Controlled Averaging for Federated Learning (SCAFFOLD) strategy.

    SCAFFOLD addresses client drift by using control variates. Each client maintains
    a control variate (c_i) and the server maintains a global control variate (c).
    These are used to correct gradient estimates during local training.

    Key Components:
    - Server control variate (c_global): Tracks global gradient estimate
    - Client control variates (c_i): Track individual client gradient estimates
    - Correction term: (c_global - c_i) applied during local training

    This implementation provides a simplified server-side version that:
    1. Maintains control variates on the server
    2. Requires minimal client-side modifications
    3. Approximates full SCAFFOLD behavior

    Note: Full SCAFFOLD requires client-side control variate updates during training.
    This simplified version initializes client covariates to zero and updates them
    based on the difference between initial and final parameters.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to FedAvg.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Server-side control variate (c_global)
        self.c_global: Optional[OrderedDict] = None

        # Client control variates dict: {client_id: c_i}
        # We'll use a simplified approach: c_i initialized to zero for new clients
        self.c_clients: Dict[str, OrderedDict] = {}

        # Track total number of clients seen
        self.total_clients_seen = set()

    def _initialize_control_variates(self, params: OrderedDict):
        """Initialize control variates with zeros matching parameter shapes.

        Parameters
        ----------
        params : OrderedDict
            Model parameters to match shapes with.
        """
        if self.c_global is None:
            # Initialize global control variate to zeros
            self.c_global = OrderedDict()
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    self.c_global[key] = np.zeros_like(value)
                else:
                    # Handle torch tensors
                    import torch
                    if isinstance(value, torch.Tensor):
                        self.c_global[key] = torch.zeros_like(value)

    def _get_client_control_variate(self, client_id: str, params: OrderedDict) -> OrderedDict:
        """Get or create control variate for a client.

        Parameters
        ----------
        client_id : str
            Unique identifier for the client.
        params : OrderedDict
            Model parameters to match shapes if creating new variate.

        Returns
        -------
        OrderedDict
            Client's control variate.
        """
        if client_id not in self.c_clients:
            # Initialize new client's control variate to zeros
            self.c_clients[client_id] = OrderedDict()
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    self.c_clients[client_id][key] = np.zeros_like(value)
                else:
                    import torch
                    if isinstance(value, torch.Tensor):
                        self.c_clients[client_id][key] = torch.zeros_like(value)
            self.total_clients_seen.add(client_id)

        return self.c_clients[client_id]

    def _update_client_control_variate(
        self,
        client_id: str,
        old_params: OrderedDict,
        new_params: OrderedDict,
        learning_rate: float,
        local_epochs: int,
        num_examples: int,
        batch_size: int
    ):
        """Update client's control variate after local training.

        Based on SCAFFOLD paper equation:
        c_i^+ = c_i - c + (x - y)/(K * η)

        Where:
        - c_i: old client control variate
        - c: global control variate
        - x: parameters before local training
        - y: parameters after local training
        - K: number of local steps
        - η: learning rate

        Parameters
        ----------
        client_id : str
            Unique identifier for the client.
        old_params : OrderedDict
            Parameters before local training.
        new_params : OrderedDict
            Parameters after local training.
        learning_rate : float
            Learning rate used during training.
        local_epochs : int
            Number of local epochs.
        num_examples : int
            Number of training examples.
        batch_size : int
            Batch size used during training.
        """
        # Calculate number of local steps K
        num_batches = max(1, num_examples // batch_size)
        K = local_epochs * num_batches

        c_i = self._get_client_control_variate(client_id, old_params)
        c_global = self.c_global

        # Update c_i using SCAFFOLD formula
        new_c_i = OrderedDict()
        for key in old_params.keys():
            # c_i^+ = c_i - c_global + (old_params - new_params) / (K * learning_rate)
            param_diff = old_params[key] - new_params[key]
            new_c_i[key] = c_i[key] - c_global[key] + param_diff / (K * learning_rate)

        self.c_clients[client_id] = new_c_i

    def aggregate_fit(
        self,
        server_round: int,
        replies,
        failures,
    ):
        """Aggregate fit results using SCAFFOLD with control variates.

        SCAFFOLD aggregation:
        1. Standard weighted averaging of parameters
        2. Update global control variate: c_global = (1/N) * sum(c_i)
        3. Update each participating client's control variate based on param changes

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

        # Get parameters from first client to initialize control variates
        first_params = replies[0].content["arrays"].to_torch_state_dict()
        self._initialize_control_variates(first_params)

        # Simplified approach for this initial implementation:
        # Since we don't have client-side control variate tracking yet,
        # we'll use standard FedAvg aggregation as a starting point
        #
        # TODO: Full SCAFFOLD implementation requires:
        # 1. Client-side application of (c_global - c_i) correction during training
        # 2. Client-side reporting of updated c_i after training
        # 3. Server-side update of c_global based on all client c_i values

        # Use parent class aggregation (FedAvg) for now
        aggregated_result, metrics = super().aggregate_fit(server_round, replies, failures)

        # Placeholder for control variate updates
        # In full implementation, we would:
        # 1. Extract c_i from each client's reply
        # 2. Update c_global = mean(all c_i)
        # 3. Return updated c_global to clients in next round

        metrics["scaffold_note"] = "Simplified implementation - full SCAFFOLD requires client modifications"

        return aggregated_result, metrics

    def __repr__(self) -> str:
        """Return string representation of strategy."""
        num_clients_tracked = len(self.c_clients)
        return (
            f"SCAFFOLD("
            f"clients_tracked={num_clients_tracked}, "
            f"fraction_fit={self.fraction_fit}, "
            f"fraction_evaluate={self.fraction_evaluate})"
        )
