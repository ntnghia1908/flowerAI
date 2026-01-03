"""Experimental ClientApp with comprehensive logging for FL experiments."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data, train as train_fn
from pytorchexample.metrics import calculate_metrics
from pytorchexample.partitioner import get_partitioner

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with comprehensive metrics."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get partition configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    distribution = context.run_config.get("distribution", "homo")
    data_source = context.run_config.get("data-source", "huggingface")

    # Get appropriate partitioner (only for HuggingFace mode)
    partitioner = get_partitioner(distribution, num_partitions) if data_source == "huggingface" else None

    # Load the data with specified partitioner or from .npy
    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size,
        partitioner=partitioner,
        data_source=data_source,
        distribution=distribution
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Calculate comprehensive metrics on training data
    train_metrics = calculate_metrics(model, trainloader, device)
    train_metrics['num_examples'] = len(trainloader.dataset)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_metrics['accuracy'],
        "train_precision": train_metrics['precision'],
        "train_recall": train_metrics['recall'],
        "train_f1": train_metrics['f1'],
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data with comprehensive metrics."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get partition configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    distribution = context.run_config.get("distribution", "homo")
    data_source = context.run_config.get("data-source", "huggingface")

    # Get appropriate partitioner (only for HuggingFace mode)
    partitioner = get_partitioner(distribution, num_partitions) if data_source == "huggingface" else None

    # Load the data with specified partitioner or from .npy
    _, valloader = load_data(
        partition_id, num_partitions, batch_size,
        partitioner=partitioner,
        data_source=data_source,
        distribution=distribution
    )

    # Calculate comprehensive metrics on validation data
    eval_metrics = calculate_metrics(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_metrics['loss'],
        "eval_acc": eval_metrics['accuracy'],
        "eval_precision": eval_metrics['precision'],
        "eval_recall": eval_metrics['recall'],
        "eval_f1": eval_metrics['f1'],
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
