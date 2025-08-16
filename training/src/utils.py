"""
Utility functions for DRL Cache Training

Provides common functionality including:
- Logging setup
- Random seed management
- Device detection
- Early stopping
- Performance monitoring
- Model checkpointing utilities
"""

import os
import sys
import random
import logging
import numpy as np
import torch
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import psutil
import json
from dataclasses import asdict

from .config import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured logger
    """
    # Create logs directory
    if config.log_to_file:
        log_dir = Path(config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, config.log_level.upper())
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)
    
    # File handler
    if config.log_to_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
    
    # Set root logger level
    logging.root.setLevel(log_level)
    
    logger = logging.getLogger('drl_cache')
    logger.info(f"Logging configured - Level: {config.log_level}, File: {config.log_file}")
    
    return logger


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get appropriate torch device.
    
    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        Torch device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    # Verify device is available
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    elif device.type == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Warning: MPS requested but not available, falling back to CPU")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    return device


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    
    Stops training when a monitored metric stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = "min", restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" - whether lower or higher is better
            restore_best_weights: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, value: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            model: Model to save best weights from
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.wait = 0
            
            # Save best weights
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() 
                                   for k, v in model.state_dict().items()}
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            
            # Restore best weights
            if model is not None and self.best_weights is not None:
                model.load_state_dict({k: v.to(model.device) 
                                     for k, v in self.best_weights.items()})
            
            return True
        
        return False


class PerformanceMonitor:
    """Monitor training performance and system resources."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
        self.system_metrics = []
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics."""
        timestamp = time.time()
        
        # Add timestamp to metrics
        metrics_with_time = {
            'timestamp': timestamp,
            'elapsed_time': timestamp - self.start_time,
            **metrics
        }
        
        self.metrics_history.append(metrics_with_time)
        
        # System metrics
        system_info = {
            'timestamp': timestamp,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                system_info.update({
                    'gpu_memory_allocated_gb': gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3),
                    'gpu_memory_reserved_gb': gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3),
                    'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                })
            except:
                pass
        
        self.system_metrics.append(system_info)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        total_time = latest['elapsed_time']
        
        # Training metrics summary
        summary = {
            'total_training_time': total_time,
            'total_epochs': len(self.metrics_history),
            'avg_epoch_time': total_time / len(self.metrics_history),
            'final_metrics': {k: v for k, v in latest.items() 
                            if k not in ['timestamp', 'elapsed_time']}
        }
        
        # System resource summary
        if self.system_metrics:
            cpu_usage = [m['cpu_percent'] for m in self.system_metrics]
            memory_usage = [m['memory_percent'] for m in self.system_metrics]
            
            summary.update({
                'avg_cpu_usage': np.mean(cpu_usage),
                'max_cpu_usage': np.max(cpu_usage),
                'avg_memory_usage': np.mean(memory_usage),
                'max_memory_usage': np.max(memory_usage)
            })
        
        return summary
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file."""
        data = {
            'training_metrics': self.metrics_history,
            'system_metrics': self.system_metrics,
            'summary': self.get_summary()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class ModelCheckpointer:
    """Utility for managing model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, keep_n_best: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_best = keep_n_best
        self.checkpoints = []
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, metrics: Dict[str, float], is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoints
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics
        })
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load(self, checkpoint_path: str, model: torch.nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', '')
        }
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the best N."""
        if len(self.checkpoints) <= self.keep_n_best:
            return
        
        # Sort by validation loss (assuming 'val_loss' key exists)
        self.checkpoints.sort(key=lambda x: x['metrics'].get('val_loss', float('inf')))
        
        # Remove worst checkpoints
        for checkpoint in self.checkpoints[self.keep_n_best:]:
            try:
                checkpoint['path'].unlink()
            except FileNotFoundError:
                pass
        
        self.checkpoints = self.checkpoints[:self.keep_n_best]


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable_parameters': trainable_params,
        'total_parameters': total_params,
        'frozen_parameters': total_params - trainable_params
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m {seconds:02d}s"
    else:
        return f"{seconds:02d}s"


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    
    return str(exp_dir)


def save_config(config: Any, filepath: str) -> None:
    """Save configuration to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if hasattr(config, '__dataclass_fields__'):
        config_dict = asdict(config)
    else:
        config_dict = config
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def test_utilities():
    """Test utility functions."""
    print("Testing DRL Cache Training Utilities")
    print("=" * 50)
    
    # Test device detection
    device = get_device("auto")
    print(f"Detected device: {device}")
    
    # Test random seed
    set_random_seed(42, deterministic=True)
    print("Random seed set to 42")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    # Simulate training loop
    val_losses = [1.0, 0.8, 0.7, 0.72, 0.71, 0.73, 0.74]
    for epoch, loss in enumerate(val_losses):
        should_stop = early_stopping(loss)
        print(f"Epoch {epoch}: loss={loss:.3f}, should_stop={should_stop}")
        if should_stop:
            break
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    
    # Simulate some metrics
    for i in range(3):
        metrics = {
            'epoch': i,
            'loss': 1.0 - i * 0.1,
            'accuracy': 0.7 + i * 0.05
        }
        monitor.update(metrics)
        time.sleep(0.1)  # Small delay
    
    summary = monitor.get_summary()
    print(f"\nPerformance summary: {summary}")
    
    # Test time formatting
    test_times = [45, 125, 3725, 7323]
    for t in test_times:
        formatted = format_time(t)
        print(f"{t}s -> {formatted}")
    
    print("\nUtility tests completed successfully!")


if __name__ == "__main__":
    test_utilities()
