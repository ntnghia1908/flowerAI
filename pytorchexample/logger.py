"""CSV logger for federated learning experiments."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ExperimentLogger:
    """Logger for FL experiments that writes metrics to CSV files."""

    def __init__(self, experiment_name: str, output_dir: str = "results"):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment (e.g., "FedAvg_Dir1.0_C5")
            output_dir: Directory to save CSV files
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this experiment run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV file paths
        self.global_csv = self.output_dir / f"{experiment_name}_global_{self.timestamp}.csv"
        self.client_csv = self.output_dir / f"{experiment_name}_client_{self.timestamp}.csv"

        # Initialize CSV files with headers
        self._init_global_csv()
        self._init_client_csv()

    def _init_global_csv(self):
        """Initialize global metrics CSV file."""
        with open(self.global_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'loss', 'accuracy', 'precision', 'recall', 'f1',
                'global_accuracy', 'weighted_accuracy'
            ])

    def _init_client_csv(self):
        """Initialize client metrics CSV file."""
        with open(self.client_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'client_id', 'phase', 'loss', 'accuracy',
                'precision', 'recall', 'f1', 'num_examples'
            ])


    def log_global_metrics(self, round_num: int, metrics: Dict[str, float]):
        """Log global evaluation metrics.

        Args:
            round_num: Current federated learning round
            metrics: Dictionary with keys: loss, accuracy, precision, recall, f1,
                     global_accuracy, weighted_accuracy
        """
        with open(self.global_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                metrics.get('loss', 0.0),
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1', 0.0),
                metrics.get('global_accuracy', 0.0),
                metrics.get('weighted_accuracy', 0.0)
            ])

    def log_client_metrics(
        self,
        round_num: int,
        client_id: int,
        phase: str,
        metrics: Dict[str, Any]
    ):
        """Log client-side metrics.

        Args:
            round_num: Current federated learning round
            client_id: ID of the client
            phase: 'train' or 'evaluate'
            metrics: Dictionary with metrics including num_examples
        """
        with open(self.client_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                client_id,
                phase,
                metrics.get('loss', 0.0),
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1', 0.0),
                metrics.get('num_examples', 0)
            ])


    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration to a separate file.

        Args:
            config: Dictionary with experiment configuration
        """
        config_file = self.output_dir / f"{self.experiment_name}_config_{self.timestamp}.txt"
        with open(config_file, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("\nConfiguration:\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

    def get_summary(self):
        """Get summary of logged data paths."""
        return {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'global_csv': str(self.global_csv),
            'client_csv': str(self.client_csv)
        }
