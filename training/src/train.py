"""
Main training script for DRL Cache

This script orchestrates the complete training pipeline:
1. Load and process access logs
2. Train dueling DQN with prioritized experience replay
3. Export model to ONNX for production deployment
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import onnx
import onnxruntime as ort

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import DRLCacheConfig, load_default_config
from model import DuelingDQN, ModelManager
from data_pipeline import DataPipeline
from replay_buffer import PrioritizedReplayBuffer
from reward_calculator import RewardCalculator
from utils import setup_logging, set_random_seed, get_device, EarlyStopping


class DRLCacheTrainer:
    """Main trainer class for DRL Cache."""
    
    def __init__(self, config: DRLCacheConfig):
        self.config = config
        self.device = get_device(config.system.device)
        
        # Setup logging
        self.logger = setup_logging(config.logging)
        
        # Set random seeds
        set_random_seed(config.system.seed, config.system.deterministic)
        
        # Initialize components
        self.model_manager = ModelManager(config.model)
        self.model = self.model_manager.create_model(use_attention=False)
        self.target_model = self.model_manager.create_model(use_attention=False)
        
        # Copy model to target
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training components
        self.replay_buffer = PrioritizedReplayBuffer(config.replay)
        self.reward_calculator = RewardCalculator(config.reward)
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.epsilon = config.training.epsilon_start
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Create output directories
        self._create_directories()
        
        self.logger.info("DRL Cache Trainer initialized")
        self.logger.info(f"Model: {self.model_manager.get_model_summary(self.model)}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.training.lr_scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.lr_scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_factor
            )
        elif self.config.training.lr_scheduler.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.training.lr_decay_factor
            )
        else:
            return None
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config.output_dir,
            self.config.logging.checkpoint_dir,
            self.config.logging.tensorboard_dir,
            "models/onnx",
            "logs",
            "data"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, log_path: str) -> Tuple[List, List, List]:
        """Load and process training data."""
        self.logger.info(f"Loading data from: {log_path}")
        
        # Create data pipeline
        pipeline = DataPipeline(self.config.data, self.config.features)
        
        # Process log file
        training_data = pipeline.process_log_file(log_path)
        
        if len(training_data) == 0:
            raise ValueError("No training data generated from log file")
        
        # Create datasets
        train_data, val_data, test_data = pipeline.create_datasets(training_data)
        
        self.logger.info(f"Data loaded - Train: {len(train_data)}, "
                        f"Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def compute_rewards(self, training_data: List[Dict]) -> List[Dict]:
        """Compute rewards for training data."""
        self.logger.info("Computing rewards...")
        
        enhanced_data = []
        for sample in tqdm(training_data, desc="Computing rewards"):
            # Calculate rewards based on future cache hits
            rewards = self.reward_calculator.compute_rewards(
                sample['candidate_keys'],
                sample['actions'],
                sample['timestamp'],
                training_data  # Full dataset for future lookup
            )
            
            sample_with_rewards = sample.copy()
            sample_with_rewards['rewards'] = rewards
            enhanced_data.append(sample_with_rewards)
        
        return enhanced_data
    
    def populate_replay_buffer(self, training_data: List[Dict]) -> None:
        """Populate replay buffer with training data."""
        self.logger.info("Populating replay buffer...")
        
        for sample in tqdm(training_data, desc="Adding to replay buffer"):
            # Convert to tensors
            features = torch.FloatTensor(sample['features']).flatten()  # [K * feature_dim]
            actions = torch.LongTensor(sample['actions'])
            rewards = torch.FloatTensor(sample['rewards'])
            
            # Add to replay buffer
            self.replay_buffer.add(
                state=features,
                actions=actions,
                rewards=rewards,
                next_state=features,  # Simplified for this implementation
                done=torch.zeros_like(rewards)
            )
        
        self.logger.info(f"Replay buffer size: {len(self.replay_buffer)}")
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.config.replay.min_size_to_sample:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.training.batch_size)
        if batch is None:
            return {}
        
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current Q-values
        current_q_values = self.model(states)
        
        # Create action mask for multi-action setup
        batch_size, max_k = actions.shape
        action_q_values = torch.sum(current_q_values * actions.float(), dim=1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards.sum(dim=1) + \
                             self.config.training.gamma * max_next_q_values * (1 - dones.sum(dim=1))
        
        # Compute loss
        td_errors = target_q_values - action_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.training.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip_norm
            )
        
        self.optimizer.step()
        
        # Update priorities in replay buffer
        if self.config.replay.prioritized:
            priorities = torch.abs(td_errors).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update epsilon
        self.epsilon = max(
            self.config.training.epsilon_end,
            self.epsilon * self.config.training.epsilon_decay
        )
        
        # Update target network
        if self.global_step % self.config.training.target_update_freq == 0:
            if self.config.training.use_soft_target_update:
                # Soft update
                tau = self.config.training.target_update_tau
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            else:
                # Hard update
                self.target_model.load_state_dict(self.model.state_dict())
        
        self.global_step += 1
        
        # Return metrics
        return {
            'loss': loss.item(),
            'q_value_mean': current_q_values.mean().item(),
            'q_value_std': current_q_values.std().item(),
            'epsilon': self.epsilon,
            'td_error_mean': td_errors.abs().mean().item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_data: List[Dict]) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for sample in val_data:
                features = torch.FloatTensor(sample['features']).flatten().unsqueeze(0).to(self.device)
                actions = torch.LongTensor(sample['actions']).unsqueeze(0).to(self.device)
                rewards = torch.FloatTensor(sample['rewards']).unsqueeze(0).to(self.device)
                
                q_values = self.model(features)
                action_q_values = torch.sum(q_values * actions.float(), dim=1)
                target_values = rewards.sum(dim=1)
                
                loss = nn.MSELoss()(action_q_values, target_values)
                total_loss += loss.item()
                total_samples += 1
        
        self.model.train()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return {'val_loss': avg_loss}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        steps_per_epoch = self.config.training.steps_per_epoch
        
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}"):
            step_metrics = self.train_step()
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Log periodically
            if step % self.config.logging.log_freq == 0 and step_metrics:
                self.logger.debug(f"Step {step}: {step_metrics}")
        
        # Average metrics over epoch
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'val_loss': val_loss,
            'epsilon': self.epsilon,
            'global_step': self.global_step,
            'replay_buffer_size': len(self.replay_buffer)
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.logging.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.logging.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the last N."""
        checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > self.config.logging.keep_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.logging.keep_n_checkpoints]:
                checkpoint.unlink()
    
    def export_to_onnx(self, output_path: str) -> None:
        """Export trained model to ONNX format."""
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Set model to eval mode
        self.model.eval()
        
        # Create dummy input
        batch_size = 1
        input_dim = self.config.model.max_k * self.config.model.input_dim
        dummy_input = torch.randn(batch_size, input_dim).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.config.export.onnx_opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=self.config.export.onnx_dynamic_axes
        )
        
        # Validate ONNX model
        if self.config.export.validate_exported_model:
            self._validate_onnx_model(output_path, dummy_input)
        
        # Optimize model
        if self.config.export.optimize_model:
            self._optimize_onnx_model(output_path)
        
        self.logger.info(f"Model exported to: {output_path}")
    
    def _validate_onnx_model(self, onnx_path: str, test_input: torch.Tensor) -> None:
        """Validate ONNX model against PyTorch model."""
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = self.model(test_input).cpu().numpy()
        
        # Get ONNX output
        onnx_output = ort_session.run(
            None,
            {'input': test_input.cpu().numpy()}
        )[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        if max_diff > self.config.export.export_test_tolerance:
            raise ValueError(f"ONNX model validation failed: max_diff={max_diff}")
        
        self.logger.info(f"ONNX model validation passed: max_diff={max_diff:.2e}")
    
    def _optimize_onnx_model(self, onnx_path: str) -> None:
        """Optimize ONNX model for inference."""
        try:
            import onnxoptimizer
            
            onnx_model = onnx.load(onnx_path)
            optimized_model = onnxoptimizer.optimize(onnx_model)
            
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            self.logger.info(f"Optimized ONNX model saved: {optimized_path}")
        
        except ImportError:
            self.logger.warning("onnxoptimizer not available, skipping optimization")
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Compute rewards and populate replay buffer
        train_data_with_rewards = self.compute_rewards(train_data)
        self.populate_replay_buffer(train_data_with_rewards)
        
        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.training.eval_freq == 0:
                val_metrics = self.validate(val_data)
                val_loss = val_metrics.get('val_loss', float('inf'))
                
                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Early stopping
                if self.early_stopping(val_loss):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Save checkpoint
                if epoch % self.config.logging.checkpoint_freq == 0:
                    self.save_checkpoint(epoch, val_loss, is_best)
                
                # Log metrics
                combined_metrics = {**train_metrics, **val_metrics}
                self.logger.info(f"Epoch {epoch}: {combined_metrics}")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        self.logger.info("Training completed!")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train DRL Cache model")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-path", type=str, required=True, help="NGINX access log path")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--export-only", action="store_true", help="Only export model to ONNX")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = DRLCacheConfig.from_yaml(args.config)
    else:
        config = load_default_config()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Validate configuration
    config.validate()
    
    # Create trainer
    trainer = DRLCacheTrainer(config)
    
    # Export only mode
    if args.export_only:
        if not args.resume:
            raise ValueError("--resume required for --export-only mode")
        
        # Load checkpoint
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Export to ONNX
        onnx_path = Path(config.output_dir) / "models" / "policy.onnx"
        trainer.export_to_onnx(str(onnx_path))
        return
    
    # Load data
    train_data, val_data, test_data = trainer.load_data(args.log_path)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.epsilon = checkpoint['epsilon']
        trainer.logger.info(f"Resumed training from epoch {checkpoint['epoch']}")
    
    # Train model
    trainer.train(train_data, val_data)
    
    # Export final model
    onnx_path = Path(config.output_dir) / "models" / "policy.onnx"
    trainer.export_to_onnx(str(onnx_path))
    
    # Test model if test data available
    if test_data:
        test_metrics = trainer.validate(test_data)
        trainer.logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
