"""Client app using pre-partitioned .npy data for faster loading."""

from flwr.client import ClientApp
from flwr.common import Context

from pytorchexample.task_experiment import Net, train, test
from pytorchexample.task_npy import load_npy_partition, get_data_dir


def train_fn(net, trainloader, epochs, lr, device):
    """Train model (wrapper for compatibility)."""
    return train(net, trainloader, epochs, lr, device)


def test_fn(net, testloader, device):
    """Test model (wrapper for compatibility)."""
    return test(net, testloader, device)


def client_fn(context: Context):
    """Client function using pre-partitioned .npy data."""
    # Get partition ID from context
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Get config
    batch_size = context.run_config.get("batch-size", 32)
    distribution = context.run_config.get("distribution", "homo")
    data_base_dir = context.run_config.get("data-dir", "./data")

    # Get data directory
    data_dir = get_data_dir(distribution, num_partitions, data_base_dir)

    print(f"Client {partition_id}: Loading data from {data_dir}")

    # Load partition data from .npy files
    trainloader, testloader = load_npy_partition(
        data_dir=data_dir,
        partition_id=partition_id,
        batch_size=batch_size
    )

    # Create model
    net = Net()

    # Import and create FlowerClient
    from pytorchexample.client_app_experiment import FlowerClient

    return FlowerClient(
        net=net,
        trainloader=trainloader,
        testloader=testloader,
        train_fn=train_fn,
        test_fn=test_fn
    ).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
