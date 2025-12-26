"""Metrics calculation utilities for federated learning experiments."""

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def calculate_metrics(net, dataloader, device, num_classes=10):
    """Calculate comprehensive metrics: loss, accuracy, precision, recall, f1.

    Args:
        net: PyTorch model
        dataloader: DataLoader for evaluation
        device: torch device (cpu or cuda)
        num_classes: Number of classes for classification

    Returns:
        dict: Dictionary containing all metrics
    """
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Store for precision, recall, f1
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate precision, recall, f1 (macro average)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_weight_metrics(current_weights, previous_weights):
    """Calculate metrics on weight changes.

    Args:
        current_weights: Current model state dict
        previous_weights: Previous model state dict

    Returns:
        dict: Dictionary containing weight change metrics
    """
    if previous_weights is None:
        return {
            'weight_norm': 0.0,
            'weight_change': 0.0,
            'weight_relative_change': 0.0
        }

    total_norm = 0.0
    total_change = 0.0
    total_prev_norm = 0.0

    for key in current_weights.keys():
        if 'weight' in key or 'bias' in key:
            curr = current_weights[key].float()
            prev = previous_weights[key].float()

            # Calculate norms
            total_norm += torch.norm(curr).item() ** 2
            total_prev_norm += torch.norm(prev).item() ** 2

            # Calculate change
            diff = curr - prev
            total_change += torch.norm(diff).item() ** 2

    weight_norm = np.sqrt(total_norm)
    weight_change = np.sqrt(total_change)
    weight_relative_change = weight_change / (np.sqrt(total_prev_norm) + 1e-10)

    return {
        'weight_norm': weight_norm,
        'weight_change': weight_change,
        'weight_relative_change': weight_relative_change
    }
