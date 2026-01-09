"""CSV logger for federated learning experiments."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import psutil


class ExperimentLogger:
    """Logger for FL experiments that writes metrics to CSV files."""

    def __init__(self, experiment_name: str, output_dir: str = "results"):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment (e.g., "FedAvg_Dir1.0_C5")
            output_dir: Directory to save CSV files
        """
        self.experiment_name = experiment_name

        # Extract strategy name for organizing results
        # e.g., "FedAvg_homo_npy" -> "FedAvg"
        self.strategy_name = experiment_name.split('_')[0]

        # Create experiment-specific subfolder: results/FedAvg_lr001_round3/
        # For now, use simple naming. Can be enhanced to include more config details
        exp_folder = f"{self.strategy_name}"
        self.output_dir = Path(output_dir) / exp_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this experiment run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV file paths
        self.global_csv = self.output_dir / f"{experiment_name}_global_{self.timestamp}.csv"
        self.client_csv = self.output_dir / f"{experiment_name}_client_{self.timestamp}.csv"
        self.hardware_csv = self.output_dir / f"{experiment_name}_hardware_{self.timestamp}.csv"

        # Track best model metrics
        self.best_accuracy = 0.0
        self.best_round = 0
        self.best_model_path: Optional[str] = None

        # Initialize CSV files with headers
        self._init_global_csv()
        self._init_client_csv()
        self._init_hardware_csv()

        # Initialize hardware monitoring
        self.process = psutil.Process()
        self.process.cpu_percent()  # First call returns 0, so call it once

        # Try to initialize GPU monitoring (NVIDIA only)
        self.has_gpu_mon = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.has_gpu_mon = True
            self.pynvml = pynvml
        except:
            pass

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

    def _init_hardware_csv(self):
        """Initialize hardware metrics CSV file."""
        with open(self.hardware_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'cpu_percent', 'ram_gb', 'gpu_temp', 'gpu_util', 'vram_gb'
            ])

    def _get_hardware_metrics(self):
        """Get current hardware utilization metrics.

        Returns:
            Dict with keys: cpu_percent, ram_gb, gpu_temp, gpu_util, vram_gb
        """
        # CPU and RAM (process-specific)
        ram_gb = self.process.memory_info().rss / (1024**3)
        cpu_percent = self.process.cpu_percent()

        # GPU metrics (system-wide if available)
        gpu_temp, gpu_util, vram_gb = 0, 0, 0
        if self.has_gpu_mon:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                gpu_temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                vram_gb = self.pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)
            except:
                pass
        # Fallback for VRAM if pynvml fails but torch CUDA is available
        elif os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            try:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.memory_reserved(0) / (1024**3)
            except:
                pass

        return {
            'cpu_percent': cpu_percent,
            'ram_gb': ram_gb,
            'gpu_temp': gpu_temp,
            'gpu_util': gpu_util,
            'vram_gb': vram_gb
        }

    def log_hardware_metrics(self, round_num: int):
        """Log hardware utilization metrics.

        Args:
            round_num: Current federated learning round
        """
        hw_metrics = self._get_hardware_metrics()
        with open(self.hardware_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                f"{hw_metrics['cpu_percent']:.1f}",
                f"{hw_metrics['ram_gb']:.2f}",
                hw_metrics['gpu_temp'],
                hw_metrics['gpu_util'],
                f"{hw_metrics['vram_gb']:.2f}"
            ])


    def log_global_metrics(self, round_num: int, metrics: Dict[str, float]):
        """Log global evaluation metrics and track best model.

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

        # Track best model based on global_accuracy
        current_acc = metrics.get('global_accuracy', 0.0)
        if current_acc > self.best_accuracy:
            self.best_accuracy = current_acc
            self.best_round = round_num

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
        """Get summary of logged data paths and best metrics."""
        return {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'global_csv': str(self.global_csv),
            'client_csv': str(self.client_csv),
            'hardware_csv': str(self.hardware_csv),
            'best_round': self.best_round,
            'best_accuracy': self.best_accuracy,
            'best_model_path': self.best_model_path
        }

    def set_best_model_path(self, model_path: str):
        """Set the path to the saved best model.

        Args:
            model_path: Path where the best model was saved
        """
        self.best_model_path = model_path
