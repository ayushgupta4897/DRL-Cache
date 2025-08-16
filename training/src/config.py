"""
DRL Cache Training Configuration

This module defines all configuration parameters for the training pipeline,
including model architecture, training hyperparameters, and system settings.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    
    # Input/output dimensions
    input_dim: int = 6  # 6 features per candidate
    max_k: int = 32     # Maximum number of candidates
    hidden_dim: int = 256
    
    # Dueling DQN architecture
    value_hidden_dim: int = 128
    advantage_hidden_dim: int = 128
    
    # Network depth
    num_hidden_layers: int = 2
    dropout_rate: float = 0.1
    activation: str = "relu"  # "relu", "gelu", "swish"
    
    # Normalization
    use_batch_norm: bool = False
    use_layer_norm: bool = True
    
    # Initialization
    weight_init: str = "xavier_uniform"  # "xavier_uniform", "he_normal", "orthogonal"


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Optimization
    learning_rate: float = 3e-4
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "step", "exponential", "none"
    lr_warmup_steps: int = 1000
    lr_decay_factor: float = 0.5
    lr_decay_steps: int = 10000
    
    # DQN specific
    gamma: float = 0.97  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Target network updates
    target_update_freq: int = 1000  # Hard update frequency
    target_update_tau: float = 0.005  # Soft update factor (if using soft updates)
    use_soft_target_update: bool = True
    
    # Training schedule
    batch_size: int = 4096
    num_epochs: int = 100
    steps_per_epoch: int = 1000
    eval_freq: int = 5  # Evaluate every N epochs
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class ReplayBufferConfig:
    """Experience replay buffer configuration."""
    
    capacity: int = 2_000_000  # 2M transitions
    prioritized: bool = True
    
    # Prioritized Experience Replay (PER)
    alpha: float = 0.6  # Prioritization strength
    beta_start: float = 0.4  # Importance sampling start
    beta_end: float = 1.0   # Importance sampling end
    epsilon: float = 1e-6   # Small constant for numerical stability
    
    # Sampling
    min_size_to_sample: int = 10000  # Minimum buffer size before sampling


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    
    # Feature names (must match C++ implementation)
    feature_names: List[str] = field(default_factory=lambda: [
        "age_sec",
        "size_kb", 
        "hit_count",
        "inter_arrival_dt",
        "ttl_left_sec",
        "last_origin_rtt_us"
    ])
    
    # Feature normalization
    normalize_features: bool = True
    normalization_method: str = "standard"  # "standard", "minmax", "robust"
    clip_outliers: bool = True
    outlier_clip_sigma: float = 5.0
    
    # Feature transformations
    log_scale_size: bool = True
    sqrt_transform_hits: bool = False
    
    # Online statistics tracking
    update_stats_online: bool = True
    stats_momentum: float = 0.999  # For exponential moving averages


@dataclass
class RewardConfig:
    """Reward function configuration."""
    
    # Base reward
    hit_reward: float = 1.0
    miss_penalty: float = 0.0
    
    # Size-proportional penalty
    use_size_penalty: bool = True
    size_penalty_lambda: float = 0.05  # Weight for size penalty
    size_penalty_scale: str = "mb"     # "kb", "mb", "log"
    
    # TTL-based rewards
    use_ttl_bonus: bool = False
    ttl_bonus_scale: float = 0.1
    
    # Frequency-based rewards  
    use_frequency_bonus: bool = True
    frequency_bonus_scale: float = 0.2


@dataclass
class SimulationConfig:
    """Cache simulation configuration."""
    
    # Cache parameters (should match production)
    max_size_gb: float = 50.0      # Cache size limit
    keys_zone_mb: float = 512.0    # Keys zone memory
    inactive_time_hours: float = 12.0  # TTL for inactive objects
    
    # Simulation settings
    warmup_ratio: float = 0.1      # Fraction of log to use for warmup
    k_candidates: int = 16         # Number of LRU tail candidates
    
    # LRU fallback behavior
    lru_fallback_ratio: float = 0.1  # Fraction of evictions to use LRU
    
    # Performance tracking
    track_hit_ratio: bool = True
    track_byte_hit_ratio: bool = True
    track_origin_offload: bool = True


@dataclass 
class DataConfig:
    """Data processing configuration."""
    
    # Input data
    log_path: str = "/var/log/nginx/access.log"
    log_format: str = "nginx_combined"  # "nginx_combined", "custom"
    
    # Data filtering
    min_object_size: int = 1024      # 1KB minimum
    max_object_size: int = 1024**3   # 1GB maximum
    min_cache_duration: int = 60     # 1 minute minimum TTL
    
    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Processing
    chunk_size: int = 10000          # Process logs in chunks
    max_workers: int = 4             # Parallel processing threads
    
    # Caching
    cache_processed_data: bool = True
    data_cache_dir: str = "./data/cache"


@dataclass
class ExportConfig:
    """Model export configuration."""
    
    # ONNX export
    onnx_opset_version: int = 11
    onnx_dynamic_axes: Dict[str, Dict[int, str]] = field(default_factory=lambda: {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    })
    
    # Model optimization
    optimize_model: bool = True
    quantize_int8: bool = True
    
    # Validation
    validate_exported_model: bool = True
    export_test_tolerance: float = 1e-4


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    
    # Basic logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "./logs/training.log"
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./logs/tensorboard"
    log_freq: int = 100  # Log every N steps
    
    # Weights & Biases (optional)
    use_wandb: bool = False
    wandb_project: str = "drl-cache"
    wandb_entity: Optional[str] = None
    
    # Model checkpointing
    checkpoint_freq: int = 5  # Save every N epochs
    keep_n_checkpoints: int = 3
    checkpoint_dir: str = "./models/checkpoints"
    
    # Metrics to track
    track_metrics: List[str] = field(default_factory=lambda: [
        "loss", "q_value_mean", "q_value_std", "hit_ratio",
        "byte_hit_ratio", "eviction_efficiency", "inference_time"
    ])


@dataclass
class SystemConfig:
    """System and performance configuration."""
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True
    
    # Memory management
    max_memory_gb: float = 8.0
    gradient_checkpointing: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Performance profiling
    profile_training: bool = False
    profile_dir: str = "./logs/profiles"


@dataclass
class DRLCacheConfig:
    """Complete configuration for DRL Cache training."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Global settings
    experiment_name: str = "drl_cache_experiment"
    output_dir: str = "./outputs"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "DRLCacheConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config object with nested dataclasses
        config = cls()
        
        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name):
                section_config = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {}
        
        for field_name in self.__dataclass_fields__:
            if field_name in ['experiment_name', 'output_dir']:
                config_dict[field_name] = getattr(self, field_name)
            else:
                section_config = getattr(self, field_name)
                if hasattr(section_config, '__dataclass_fields__'):
                    config_dict[field_name] = {}
                    for sub_field in section_config.__dataclass_fields__:
                        config_dict[field_name][sub_field] = getattr(section_config, sub_field)
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Model validation
        assert self.model.input_dim > 0, "Input dimension must be positive"
        assert self.model.max_k > 0, "Max K must be positive"
        assert self.model.hidden_dim > 0, "Hidden dimension must be positive"
        
        # Training validation
        assert 0 < self.training.learning_rate < 1, "Learning rate must be in (0, 1)"
        assert 0 < self.training.gamma <= 1, "Gamma must be in (0, 1]"
        assert self.training.batch_size > 0, "Batch size must be positive"
        
        # Replay buffer validation
        assert self.replay.capacity > 0, "Replay buffer capacity must be positive"
        if self.replay.prioritized:
            assert 0 <= self.replay.alpha <= 1, "Alpha must be in [0, 1]"
            assert 0 <= self.replay.beta_start <= 1, "Beta start must be in [0, 1]"
            assert 0 <= self.replay.beta_end <= 1, "Beta end must be in [0, 1]"
        
        # Data validation
        assert 0 < self.data.train_ratio < 1, "Train ratio must be in (0, 1)"
        assert 0 < self.data.val_ratio < 1, "Val ratio must be in (0, 1)"
        assert 0 < self.data.test_ratio < 1, "Test ratio must be in (0, 1)"
        assert abs(self.data.train_ratio + self.data.val_ratio + self.data.test_ratio - 1.0) < 1e-6, \
            "Data split ratios must sum to 1"
        
        # System validation
        assert self.system.num_workers >= 0, "Number of workers must be non-negative"
        assert self.system.max_memory_gb > 0, "Max memory must be positive"
        
        print("✓ Configuration validation passed")


def load_default_config() -> DRLCacheConfig:
    """Load default configuration."""
    return DRLCacheConfig()


def create_sample_config(output_path: str) -> None:
    """Create a sample configuration file."""
    config = load_default_config()
    config.to_yaml(output_path)
    print(f"Sample configuration created at: {output_path}")


if __name__ == "__main__":
    # Create sample configuration
    sample_path = "./config/default.yaml"
    create_sample_config(sample_path)
    
    # Test loading
    config = DRLCacheConfig.from_yaml(sample_path)
    config.validate()
    print("Configuration system working correctly!")
