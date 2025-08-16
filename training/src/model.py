"""
Dueling Deep Q-Network (DQN) Implementation for DRL Cache

This module implements the dueling DQN architecture specifically designed for 
cache eviction decisions. The model takes features for K candidate objects
and outputs Q-values for the eviction action of each candidate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .config import ModelConfig


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for cache eviction decisions.
    
    Architecture:
    - Shared feature extraction layers
    - Separate value and advantage streams
    - Dueling combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    
    Args:
        config: Model configuration parameters
    """
    
    def __init__(self, config: ModelConfig):
        super(DuelingDQN, self).__init__()
        self.config = config
        self.input_dim = config.input_dim * config.max_k  # Flattened features
        self.max_k = config.max_k
        
        # Feature extraction (shared layers)
        self.feature_extractor = self._build_feature_extractor()
        
        # Value stream: V(s) - single output
        self.value_stream = self._build_value_stream()
        
        # Advantage stream: A(s,a) - one output per action (candidate)
        self.advantage_stream = self._build_advantage_stream()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build shared feature extraction layers."""
        layers = []
        
        # Input layer
        in_dim = self.input_dim
        out_dim = self.config.hidden_dim
        
        layers.append(nn.Linear(in_dim, out_dim))
        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        elif self.config.use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(self._get_activation())
        
        if self.config.dropout_rate > 0:
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers
        for _ in range(self.config.num_hidden_layers - 1):
            layers.append(nn.Linear(out_dim, out_dim))
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            elif self.config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(self._get_activation())
            
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _build_value_stream(self) -> nn.Module:
        """Build value function stream V(s)."""
        layers = []
        
        # Value-specific layers
        layers.append(nn.Linear(self.config.hidden_dim, self.config.value_hidden_dim))
        if self.config.use_layer_norm:
            layers.append(nn.LayerNorm(self.config.value_hidden_dim))
        layers.append(self._get_activation())
        
        if self.config.dropout_rate > 0:
            layers.append(nn.Dropout(self.config.dropout_rate * 0.5))  # Reduced dropout
        
        # Output layer (single value)
        layers.append(nn.Linear(self.config.value_hidden_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _build_advantage_stream(self) -> nn.Module:
        """Build advantage function stream A(s,a)."""
        layers = []
        
        # Advantage-specific layers
        layers.append(nn.Linear(self.config.hidden_dim, self.config.advantage_hidden_dim))
        if self.config.use_layer_norm:
            layers.append(nn.LayerNorm(self.config.advantage_hidden_dim))
        layers.append(self._get_activation())
        
        if self.config.dropout_rate > 0:
            layers.append(nn.Dropout(self.config.dropout_rate * 0.5))  # Reduced dropout
        
        # Output layer (one advantage per candidate)
        layers.append(nn.Linear(self.config.advantage_hidden_dim, self.config.max_k))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # Swish = SiLU in PyTorch
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        return activations.get(self.config.activation.lower(), nn.ReLU())
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "he_normal":
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif self.config.weight_init == "orthogonal":
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, k_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the dueling DQN.
        
        Args:
            x: Input features [batch_size, max_k * input_dim]
            k_mask: Optional mask for valid candidates [batch_size, max_k]
                   True for valid candidates, False for padding
        
        Returns:
            Q-values for each candidate [batch_size, max_k]
        """
        batch_size = x.size(0)
        
        # Extract shared features
        features = self.feature_extractor(x)  # [batch_size, hidden_dim]
        
        # Value stream: V(s)
        value = self.value_stream(features)  # [batch_size, 1]
        
        # Advantage stream: A(s,a)
        advantage = self.advantage_stream(features)  # [batch_size, max_k]
        
        # Apply mask to advantages if provided
        if k_mask is not None:
            # Set advantage to large negative value for invalid candidates
            advantage = advantage.masked_fill(~k_mask, -1e8)
        
        # Dueling combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        # This ensures that V(s) represents the expected value over all actions
        if k_mask is not None:
            # Compute mean only over valid candidates
            valid_advantages = advantage.masked_fill(~k_mask, 0)
            valid_count = k_mask.sum(dim=1, keepdim=True).float()
            advantage_mean = valid_advantages.sum(dim=1, keepdim=True) / valid_count.clamp(min=1)
        else:
            advantage_mean = advantage.mean(dim=1, keepdim=True)
        
        q_values = value + advantage - advantage_mean
        
        # Apply mask to final Q-values if provided
        if k_mask is not None:
            q_values = q_values.masked_fill(~k_mask, -1e8)
        
        return q_values
    
    def get_action_probs(self, x: torch.Tensor, k_mask: Optional[torch.Tensor] = None, 
                        temperature: float = 1.0) -> torch.Tensor:
        """
        Get action probabilities using softmax over Q-values.
        
        Args:
            x: Input features [batch_size, max_k * input_dim]
            k_mask: Optional mask for valid candidates
            temperature: Temperature for softmax (higher = more exploration)
        
        Returns:
            Action probabilities [batch_size, max_k]
        """
        q_values = self.forward(x, k_mask) / temperature
        
        if k_mask is not None:
            # Apply mask before softmax
            q_values = q_values.masked_fill(~k_mask, -1e8)
        
        probs = F.softmax(q_values, dim=1)
        
        return probs
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance using gradients.
        
        Args:
            x: Input features [batch_size, max_k * input_dim]
        
        Returns:
            Dictionary of feature importance metrics
        """
        x.requires_grad_(True)
        
        # Forward pass
        q_values = self.forward(x)
        
        # Compute gradients w.r.t. input
        target = q_values.sum()
        gradients = torch.autograd.grad(target, x, create_graph=False)[0]
        
        # Feature importance as absolute gradients
        importance = torch.abs(gradients)
        
        # Reshape to [batch_size, max_k, input_dim]
        importance = importance.view(-1, self.max_k, self.config.input_dim)
        
        # Average over batch and candidates
        feature_importance = importance.mean(dim=(0, 1))  # [input_dim]
        
        return {
            "feature_importance": feature_importance,
            "gradients": gradients,
            "importance_per_candidate": importance
        }


class DuelingDQNWithAttention(DuelingDQN):
    """
    Enhanced Dueling DQN with attention mechanism for better candidate comparison.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Attention mechanism
        self.attention_dim = config.hidden_dim // 4
        self.candidate_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Projection layers
        self.feature_to_attention = nn.Linear(config.input_dim, self.attention_dim)
        self.attention_to_feature = nn.Linear(self.attention_dim, config.input_dim)
    
    def forward(self, x: torch.Tensor, k_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        batch_size = x.size(0)
        
        # Reshape input to [batch_size, max_k, input_dim]
        x_candidates = x.view(batch_size, self.max_k, self.config.input_dim)
        
        # Apply attention across candidates
        x_att = self.feature_to_attention(x_candidates)  # [batch_size, max_k, attention_dim]
        
        # Self-attention across candidates
        if k_mask is not None:
            # Convert mask for attention (False = ignore in attention)
            att_mask = ~k_mask  # [batch_size, max_k]
        else:
            att_mask = None
        
        attended, _ = self.candidate_attention(x_att, x_att, x_att, key_padding_mask=att_mask)
        attended = self.attention_to_feature(attended)  # [batch_size, max_k, input_dim]
        
        # Residual connection
        x_enhanced = x_candidates + attended
        
        # Flatten back for standard DQN processing
        x_flat = x_enhanced.view(batch_size, -1)
        
        # Continue with standard dueling DQN
        return super().forward(x_flat, k_mask)


class ModelManager:
    """Utility class for model management and operations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model(self, use_attention: bool = False) -> DuelingDQN:
        """Create a new model instance."""
        if use_attention:
            model = DuelingDQNWithAttention(self.config)
        else:
            model = DuelingDQN(self.config)
        
        return model.to(self.device)
    
    def count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Count trainable and total parameters."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params
    
    def get_model_summary(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        trainable, total = self.count_parameters(model)
        
        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "model_size_mb": model_size_mb,
            "device": str(self.device),
            "architecture": type(model).__name__
        }
    
    def save_model(self, model: nn.Module, path: str, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: int = 0, loss: float = 0.0) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "loss": loss,
            "model_class": type(model).__name__
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_model(self, path: str, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if model is None:
            # Create model from checkpoint config
            model_class_name = checkpoint.get("model_class", "DuelingDQN")
            if model_class_name == "DuelingDQNWithAttention":
                model = DuelingDQNWithAttention(checkpoint["config"])
            else:
                model = DuelingDQN(checkpoint["config"])
            model = model.to(self.device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model, checkpoint


def test_model():
    """Test the model implementation."""
    from .config import ModelConfig
    
    config = ModelConfig()
    manager = ModelManager(config)
    
    # Test standard dueling DQN
    model = manager.create_model(use_attention=False)
    print("Standard Dueling DQN:")
    print(manager.get_model_summary(model))
    
    # Test with attention
    model_att = manager.create_model(use_attention=True)
    print("\nDueling DQN with Attention:")
    print(manager.get_model_summary(model_att))
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.max_k * config.input_dim)
    k_mask = torch.randint(0, 2, (batch_size, config.max_k)).bool()
    
    with torch.no_grad():
        q_values = model(x, k_mask)
        probs = model.get_action_probs(x, k_mask)
    
    print(f"\nForward pass test:")
    print(f"Input shape: {x.shape}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Action probs shape: {probs.shape}")
    print(f"Q-values range: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")
    print(f"Probs sum: {probs.sum(dim=1).mean().item():.3f}")


if __name__ == "__main__":
    test_model()
