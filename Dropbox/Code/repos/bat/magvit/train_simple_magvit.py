#!/usr/bin/env python3
"""
Training Script for Simple MAGVIT Model
Multi-task training on synthetic video data with frame prediction, interpolation, inpainting, and outpainting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except (ImportError, AttributeError):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, AttributeError):
        # Fallback: create a dummy writer if tensorboard fails
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass
            def add_scalar(self, *args, **kwargs):
                pass
            def close(self):
                pass
import numpy as np
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from dataclasses import asdict

from simple_magvit_model import (
    SimpleMagvitModel, ModelConfig, SyntheticVideoDataset, 
    create_data_loaders, visualize_results
)

class MultiTaskVideoLoss(nn.Module):
    """Multi-task loss for MAGVIT-style video modeling"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss = nn.MSELoss()
        
        # Task weights (can be tuned)
        self.task_weights = {
            'frame_prediction': 1.0,
            'frame_interpolation': 1.0,
            'video_inpainting': 1.0,
            'video_outpainting': 1.0,
            'unconditional': 0.5  # Lower weight for unconditional generation
        }
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor, mask: torch.Tensor,
                task_type: str = 'unconditional') -> Dict[str, Union[torch.Tensor, float]]:
        """Compute multi-task loss"""
        
        # Get predicted logits
        logits = predictions['logits']  # (B, N, vocab_size)
        
        # Tokenize the TARGET video (full video, not masked input)
        # This is critical: we predict tokens for the full video, not the masked input
        target_embeddings, target_tokens = predictions['model'].tokenizer(targets)
        
        # Masked language modeling loss (only on masked positions)
        mlm_targets = target_tokens.clone()
        mlm_targets[~mask] = -100  # Ignore unmasked positions
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), mlm_targets.view(-1))
        
        # Task weight
        weight = self.task_weights.get(task_type, 1.0)
        total_loss = weight * ce_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'task_weight': weight,
            'num_masked': mask.sum().item()
        }

class VideoTrainer:
    """Trainer for Simple MAGVIT model"""
    
    def __init__(self, config: ModelConfig, args):
        self.config = config
        self.args = args
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Create model
        self.model = SimpleMagvitModel(config).to(self.device)
        self.loss_fn = MultiTaskVideoLoss(config)
        
        # Create optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
        )
        
        # Create data loaders
        self.loaders, self.datasets = create_data_loaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
        
        # Setup logging
        self.setup_logging()
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📊 Dataset: {len(self.datasets['train'])} train, {len(self.datasets['val'])} val")
    
    def setup_logging(self):
        """Setup logging and checkpointing"""
        # Create output directory
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Weights & Biases (optional)
        if self.args.use_wandb:
            wandb.init(
                project="simple-magvit",
                config=asdict(self.config),
                name=f"magvit-{self.args.run_name}"
            )
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Task types for multi-task training (including masked prediction)
        task_types = ['frame_prediction', 'frame_interpolation', 'video_inpainting', 'video_outpainting', 'frame_prediction_masked']
        
        epoch_losses = {task: [] for task in task_types + ['total']}
        
        pbar = tqdm(self.loaders['train'], desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Sample a task for this batch
            task_type = np.random.choice(task_types)
            
            # Get video data
            videos = batch['video'].to(self.device)  # (B, T, H, W, C)
            
            # Forward pass
            try:
                # For masked tasks, use the task input if available
                if task_type in batch['tasks'] and 'input' in batch['tasks'][task_type]:
                    task_videos = batch['tasks'][task_type]['input'].to(self.device)
                else:
                    task_videos = videos
                
                predictions = self.model(task_videos, task_type=task_type)
                
                # Use target video for loss if available, otherwise use full video
                if task_type in batch['tasks'] and 'target' in batch['tasks'][task_type]:
                    target_videos = batch['tasks'][task_type]['target'].to(self.device)
                else:
                    target_videos = videos
                
                # Compute loss - pass model reference for tokenization
                # Store model reference in predictions temporarily
                predictions['model'] = self.model
                loss_dict = self.loss_fn(predictions, target_videos, predictions['mask'], task_type)
                predictions.pop('model')  # Remove temporary reference
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.optimizer.step()
                
                # Record losses
                epoch_losses[task_type].append(loss_dict['ce_loss'].item())
                epoch_losses['total'].append(loss_dict['total_loss'].item())
                
                # Update progress bar
                pbar.set_postfix({
                    'task': task_type[:4],
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to tensorboard
                global_step = epoch * len(self.loaders['train']) + batch_idx
                self.writer.add_scalar(f'train/loss_{task_type}', loss_dict['ce_loss'].item(), global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
                
            except Exception as e:
                print(f"❌ Error in training step: {e}")
                continue
        
        # Calculate epoch averages
        epoch_avg = {}
        for task in epoch_losses:
            if epoch_losses[task]:
                epoch_avg[task] = np.mean(epoch_losses[task])
        
        return epoch_avg
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        task_types = ['frame_prediction', 'frame_interpolation', 'video_inpainting', 'video_outpainting']
        val_losses = {task: [] for task in task_types + ['total']}
        
        with torch.no_grad():
            for batch in tqdm(self.loaders['val'], desc='Validation', leave=False):
                videos = batch['video'].to(self.device)
                
                # Test all tasks
                for task_type in task_types:
                    try:
                        predictions = self.model(videos, task_type=task_type)
                        # Pass model reference for tokenization
                        predictions['model'] = self.model
                        loss_dict = self.loss_fn(predictions, videos, predictions['mask'], task_type)
                        predictions.pop('model')  # Remove temporary reference
                        
                        val_losses[task_type].append(loss_dict['ce_loss'].item())
                        val_losses['total'].append(loss_dict['total_loss'].item())
                        
                    except Exception as e:
                        print(f"❌ Error in validation: {e}")
                        continue
        
        # Calculate averages
        val_avg = {}
        for task in val_losses:
            if val_losses[task]:
                val_avg[task] = np.mean(val_losses[task])
        
        # Log validation metrics
        for task, loss in val_avg.items():
            self.writer.add_scalar(f'val/loss_{task}', loss, epoch)
        
        return val_avg
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> int:
        """Load checkpoint and resume training state
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, tries to load latest checkpoint.
        
        Returns:
            Starting epoch number (0 if no checkpoint found)
        """
        if checkpoint_path is None:
            # Try to load latest checkpoint
            checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
            if not checkpoint_path.exists():
                print("📭 No checkpoint found, starting from scratch")
                return 0
        
        if isinstance(checkpoint_path, Path):
            checkpoint_path = str(checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            print(f"📭 Checkpoint not found at {checkpoint_path}, starting from scratch")
            return 0
        
        print(f"📦 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load best validation loss
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Get starting epoch
        start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
        
        print(f"✅ Checkpoint loaded! Resuming from epoch {start_epoch+1}")
        print(f"   Previous best validation loss: {self.best_val_loss:.4f}")
        
        return start_epoch
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify directory was created
        if not self.output_dir.exists():
            raise RuntimeError(f"Failed to create output directory: {self.output_dir}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Verify checkpoint was saved
        if not latest_path.exists():
            raise RuntimeError(f"Failed to save checkpoint: {latest_path}")
        
        checkpoint_size = latest_path.stat().st_size / (1024 * 1024)  # MB
        if checkpoint_size < 1:
            print(f"⚠️  Warning: Checkpoint size is suspiciously small: {checkpoint_size:.1f} MB")
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            
            # Verify best checkpoint was saved
            if not best_path.exists():
                raise RuntimeError(f"Failed to save best checkpoint: {best_path}")
            
            print(f"💾 Best model saved at epoch {epoch+1} ({checkpoint_size:.1f} MB)")
        else:
            print(f"💾 Checkpoint saved at epoch {epoch+1} ({checkpoint_size:.1f} MB)")
    
    def create_visualizations(self, epoch: int):
        """Create and save visualizations"""
        self.model.eval()
        
        # Generate samples
        fig = visualize_results(self.model, self.datasets['val'], self.device, num_samples=4)
        
        # Save figure
        fig.savefig(self.output_dir / f'samples_epoch_{epoch+1:03d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Log to wandb if enabled
        if self.args.use_wandb:
            wandb.log({"generated_samples": wandb.Image(str(self.output_dir / f'samples_epoch_{epoch+1:03d}.png'))})
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop
        
        Args:
            resume_from: Path to checkpoint to resume from. If None, auto-detects latest checkpoint.
        """
        print("🚀 Starting training...")
        
        # Try to load checkpoint and resume
        start_epoch = self.load_checkpoint(resume_from)
        
        for epoch in range(start_epoch, self.args.epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate_epoch(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Track best model
            current_val_loss = val_losses.get('total', float('inf'))
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Create visualizations
            if (epoch + 1) % self.args.vis_freq == 0:
                self.create_visualizations(epoch)
            
            # Store losses
            self.train_losses.append(train_losses.get('total', 0))
            self.val_losses.append(current_val_loss)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.args.epochs} Summary:")
            print(f"   Train Loss: {train_losses.get('total', 0):.4f}")
            print(f"   Val Loss: {current_val_loss:.4f}")
            print(f"   Best Val Loss: {self.best_val_loss:.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_losses.get('total', 0),
                    'val_loss': current_val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        print("✅ Training completed!")
        
        # Final visualization
        self.create_visualizations(self.args.epochs - 1)
        
        # Save final model
        self.save_checkpoint(self.args.epochs - 1)
        
        # Plot loss curves
        self.plot_loss_curves()
    
    def plot_loss_curves(self):
        """Plot and save loss curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.axhline(y=self.best_val_loss, color='r', linestyle='--', label=f'Best: {self.best_val_loss:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Simple MAGVIT model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./synthetic_dataset',
                        help='Path to synthetic dataset')
    parser.add_argument('--output_dir', type=str, default='./experiments/simple_magvit',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from (default: auto-detect latest)')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    
    # Logging arguments
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint frequency')
    parser.add_argument('--vis_freq', type=int, default=5,
                        help='Visualization frequency')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default='default',
                        help='Run name for logging')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model config
    config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print("❌ Dataset not found. Please run synthetic_video_generator.py first!")
        return
    
    # Create trainer and start training
    trainer = VideoTrainer(config, args)
    trainer.train(resume_from=args.resume_from)

if __name__ == "__main__":
    main() 