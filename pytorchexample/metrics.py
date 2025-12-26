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
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf detected in loss calculation")
                continue

            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Store for precision, recall, f1
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics with safety checks
    dataset_size = len(dataloader.dataset)
    accuracy = correct / dataset_size if dataset_size > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate precision, recall, f1 (macro average)
    # Use zero_division=0 to avoid warnings
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    return {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
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
        # Calculate only norm for first round
        total_norm = 0.0
        for key in current_weights.keys():
            if 'weight' in key or 'bias' in key:
                curr = current_weights[key].float()
                total_norm += torch.norm(curr).item() ** 2

        return {
            'weight_norm': float(np.sqrt(total_norm)),
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
            curr_norm = torch.norm(curr).item()
            prev_norm = torch.norm(prev).item()

            # Check for NaN/Inf
            if np.isnan(curr_norm) or np.isinf(curr_norm):
                print(f"Warning: NaN/Inf detected in current weights for {key}")
                continue

            if np.isnan(prev_norm) or np.isinf(prev_norm):
                print(f"Warning: NaN/Inf detected in previous weights for {key}")
                continue

            total_norm += curr_norm ** 2
            total_prev_norm += prev_norm ** 2

            # Calculate change
            diff = curr - prev
            diff_norm = torch.norm(diff).item()

            if not (np.isnan(diff_norm) or np.isinf(diff_norm)):
                total_change += diff_norm ** 2

    weight_norm = float(np.sqrt(total_norm))
    weight_change = float(np.sqrt(total_change))

    # Avoid division by zero
    if total_prev_norm > 1e-10:
        weight_relative_change = weight_change / np.sqrt(total_prev_norm)
    else:
        weight_relative_change = 0.0

    return {
        'weight_norm': float(weight_norm),
        'weight_change': float(weight_change),
        'weight_relative_change': float(weight_relative_change)
    }
