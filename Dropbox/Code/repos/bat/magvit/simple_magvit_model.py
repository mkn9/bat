#!/usr/bin/env python3
"""
Simple MAGVIT-Style Foundation Model
A simplified implementation of MAGVIT for multi-task video learning on synthetic data.

Key components:
- Video tokenization (2D patches for simplicity)
- Transformer with masked modeling
- Multi-task training (prediction, interpolation, inpainting, outpainting)
- COMMIT-style conditional masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class ModelConfig:
    """Configuration for Simple MAGVIT model"""
    # Video dimensions
    video_height: int = 64
    video_width: int = 64
    video_frames: int = 16
    video_channels: int = 3
    
    # Tokenization
    patch_size: int = 8  # 8x8 patches
    vocab_size: int = 1024  # Token vocabulary size
    
    # Transformer
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    
    # Training
    mask_ratio: float = 0.75  # Default mask ratio
    max_mask_ratio: float = 0.95  # Max mask ratio (like VideoMAE)
    
    # Tasks
    task_embedding_dim: int = 32
    
    def __post_init__(self):
        # Calculate patch dimensions
        self.patches_h = self.video_height // self.patch_size
        self.patches_w = self.video_width // self.patch_size
        self.patches_per_frame = self.patches_h * self.patches_w
        self.total_patches = self.patches_per_frame * self.video_frames
        self.patch_dim = self.patch_size * self.patch_size * self.video_channels

class VideoTokenizer(nn.Module):
    """Simple video tokenizer using 2D patches"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = nn.Linear(config.patch_dim, config.hidden_dim)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.total_patches, config.hidden_dim)
        )
        
        # Temporal embeddings
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, config.video_frames, config.hidden_dim)
        )
        
        # Quantization (simplified - just learnable embeddings)
        self.quantize = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.quantize_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        
    def patchify(self, videos: torch.Tensor) -> torch.Tensor:
        """Convert video to patches"""
        # videos: (B, T, H, W, C)
        B, T, H, W, C = videos.shape
        
        # Reshape to patches
        videos = videos.view(
            B, T, 
            H // self.config.patch_size, self.config.patch_size,
            W // self.config.patch_size, self.config.patch_size,
            C
        )
        
        # Rearrange to get patches
        videos = videos.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        videos = videos.view(B, T * self.config.patches_per_frame, self.config.patch_dim)
        
        return videos
    
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert patches back to video"""
        # patches: (B, T*patches_per_frame, patch_dim)
        B = patches.shape[0]
        
        patches = patches.view(
            B, self.config.video_frames, 
            self.config.patches_h, self.config.patches_w,
            self.config.patch_size, self.config.patch_size,
            self.config.video_channels
        )
        
        patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        videos = patches.view(
            B, self.config.video_frames,
            self.config.video_height, self.config.video_width,
            self.config.video_channels
        )
        
        return videos
    
    def forward(self, videos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize video"""
        # Patchify
        patches = self.patchify(videos)  # (B, total_patches, patch_dim)
        
        # Embed patches
        embeddings = self.patch_embedding(patches)  # (B, total_patches, hidden_dim)
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embedding
        
        # Add temporal embeddings
        temp_emb = self.temporal_embedding.repeat_interleave(
            self.config.patches_per_frame, dim=1
        )
        embeddings = embeddings + temp_emb
        
        # Quantize (simplified - just get token indices)
        logits = self.quantize_proj(embeddings)  # (B, total_patches, vocab_size)
        token_ids = torch.argmax(logits, dim=-1)  # (B, total_patches)
        
        return embeddings, token_ids
    
    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decode token IDs back to video"""
        # Get embeddings from token IDs
        embeddings = self.quantize(token_ids)  # (B, total_patches, hidden_dim)
        
        # Project back to patch dimension using inverse linear transformation
        # Create inverse projection layer if not exists
        if not hasattr(self, 'patch_decode'):
            self.patch_decode = nn.Linear(self.config.hidden_dim, self.config.patch_dim)
            # Initialize with transpose of patch_embedding weights
            with torch.no_grad():
                self.patch_decode.weight.data = self.patch_embedding.weight.data.T
                if self.patch_decode.bias is not None:
                    self.patch_decode.bias.data.zero_()
            # Move to same device as embeddings
            self.patch_decode = self.patch_decode.to(embeddings.device)
        
        patches = self.patch_decode(embeddings)  # (B, total_patches, patch_dim)
        
        # Unpatchify
        videos = self.unpatchify(patches)
        
        # Apply sigmoid normalization to ensure [0, 1] range
        videos = torch.sigmoid(videos)
        
        return videos

class MaskedVideoTransformer(nn.Module):
    """Transformer for masked video modeling"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(5, config.task_embedding_dim)  # 5 tasks
        self.task_projection = nn.Linear(config.task_embedding_dim, config.hidden_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * config.mlp_ratio,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, 
                task_id: int = 0) -> torch.Tensor:
        """Forward pass with masked modeling"""
        B, N, D = embeddings.shape
        
        # Add task embedding
        task_emb = self.task_embeddings(torch.tensor(task_id, device=embeddings.device))
        task_emb = self.task_projection(task_emb).unsqueeze(0).expand(B, -1, -1)
        
        # Apply mask
        mask_tokens = self.mask_token.expand(B, N, -1)
        masked_embeddings = torch.where(mask.unsqueeze(-1), mask_tokens, embeddings)
        
        # Add task embedding to first position
        masked_embeddings = torch.cat([task_emb, masked_embeddings], dim=1)
        
        # Transformer
        output = self.transformer(masked_embeddings)
        
        # Remove task token and project
        output = output[:, 1:]  # Remove task token
        logits = self.output_projection(output)
        
        return logits

class SimpleMagvitModel(nn.Module):
    """Complete Simple MAGVIT model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tokenizer = VideoTokenizer(config)
        self.transformer = MaskedVideoTransformer(config)
        
        # Task mapping
        self.task_map = {
            'frame_prediction': 0,
            'frame_interpolation': 1,
            'video_inpainting': 2,
            'video_outpainting': 3,
            'unconditional': 4
        }
        
    def create_mask(self, batch_size: int, sequence_length: int, 
                   mask_ratio: Optional[float] = None) -> torch.Tensor:
        """Create random mask for training"""
        if mask_ratio is None:
            mask_ratio = random.uniform(0.5, self.config.max_mask_ratio)
        
        num_masked = int(sequence_length * mask_ratio)
        
        mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
        for b in range(batch_size):
            masked_indices = torch.randperm(sequence_length)[:num_masked]
            mask[b, masked_indices] = True
            
        return mask
    
    def forward(self, videos: torch.Tensor, task_type: str = 'unconditional',
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Tokenize
        embeddings, token_ids = self.tokenizer(videos)
        
        # Create mask if not provided
        if mask is None:
            mask = self.create_mask(videos.shape[0], embeddings.shape[1])
            
        # Get task ID
        task_id = self.task_map.get(task_type, 4)
        
        # Forward through transformer
        logits = self.transformer(embeddings, mask.to(embeddings.device), task_id)
        
        return {
            'logits': logits,
            'token_ids': token_ids,
            'mask': mask,
            'embeddings': embeddings
        }
    
    def generate(self, condition_videos: torch.Tensor, task_type: str,
                 num_steps: int = 12) -> torch.Tensor:
        """Generate video using iterative refinement"""
        self.eval()
        
        with torch.no_grad():
            # Initial tokenization
            embeddings, _ = self.tokenizer(condition_videos)
            B, N, D = embeddings.shape
            
            # Start with all masked
            current_tokens = torch.zeros(B, N, dtype=torch.long, device=embeddings.device)
            mask = torch.ones(B, N, dtype=torch.bool, device=embeddings.device)
            
            for step in range(num_steps):
                # Current mask ratio
                mask_ratio = 1.0 - (step + 1) / num_steps
                
                # Forward pass
                task_id = self.task_map.get(task_type, 4)
                logits = self.transformer(embeddings, mask, task_id)
                
                # Sample tokens
                probs = F.softmax(logits, dim=-1)
                sampled_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
                sampled_tokens = sampled_tokens.view(B, N)
                
                # Update tokens where masked
                current_tokens = torch.where(mask, sampled_tokens, current_tokens)
                
                # Update mask for next iteration
                if step < num_steps - 1:
                    num_keep = int(N * (1 - mask_ratio))
                    # Keep tokens with highest confidence
                    confidence = torch.max(probs, dim=-1)[0]
                    _, keep_indices = torch.topk(confidence, num_keep, dim=1)
                    
                    new_mask = torch.ones(B, N, dtype=torch.bool, device=embeddings.device)
                    for b in range(B):
                        new_mask[b, keep_indices[b]] = False
                    mask = new_mask
            
            # Decode final tokens
            generated_videos = self.tokenizer.decode(current_tokens)
            
        return generated_videos

class SyntheticVideoDataset(Dataset):
    """Dataset for synthetic video data"""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load split
        with open(self.data_dir / f'{split}_split.txt', 'r') as f:
            self.video_ids = [line.strip() for line in f.readlines()]
        
        # Load dataset metadata
        with open(self.data_dir / 'dataset_metadata.json', 'r') as f:
            self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load video
        video_path = self.data_dir / 'videos' / f'{video_id}.npy'
        video = np.load(video_path)  # (T, H, W, C)
        
        # Load tasks
        tasks = {}
        task_types = ['frame_prediction', 'frame_interpolation', 'video_inpainting', 'video_outpainting', 'frame_prediction_masked']
        
        for task_type in task_types:
            input_path = self.data_dir / 'tasks' / f'{video_id}_{task_type}_input.npy'
            target_path = self.data_dir / 'tasks' / f'{video_id}_{task_type}_target.npy'
            mask_path = self.data_dir / 'tasks' / f'{video_id}_{task_type}_mask.npy'
            
            if input_path.exists() and target_path.exists():
                task_input = np.load(input_path)
                task_target = np.load(target_path)
                
                task_data = {
                    'input': torch.from_numpy(task_input).float() / 255.0,
                    'target': torch.from_numpy(task_target).float() / 255.0
                }
                
                # Load mask if available
                if mask_path.exists():
                    mask = np.load(mask_path)
                    task_data['mask'] = torch.from_numpy(mask).bool()
                
                tasks[task_type] = task_data
        
        return {
            'video_id': video_id,
            'video': torch.from_numpy(video).float() / 255.0,  # Normalize to [0, 1]
            'tasks': tasks
        }

def create_data_loaders(data_dir: str, batch_size: int = 16, num_workers: int = 4):
    """Create data loaders for training"""
    datasets = {}
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = SyntheticVideoDataset(data_dir, split)
        loaders[split] = DataLoader(
            datasets[split], 
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders, datasets

def visualize_results(model: SimpleMagvitModel, dataset: SyntheticVideoDataset, 
                     device: torch.device, num_samples: int = 2):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        sample = dataset[i]
        video = sample['video'].unsqueeze(0).to(device)  # Add batch dim
        
        # Test frame prediction
        if 'frame_prediction' in sample['tasks']:
            with torch.no_grad():
                result = model(video, 'frame_prediction')
                generated = model.generate(video, 'frame_prediction', num_steps=8)
            
            # Show original, input, target, generated
            axes[i, 0].imshow(video[0, 0].cpu().numpy())
            axes[i, 0].set_title('Original Frame 0')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(video[0, 8].cpu().numpy())
            axes[i, 1].set_title('Original Frame 8')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(generated[0, 8].cpu().numpy())
            axes[i, 2].set_title('Generated Frame 8')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(video[0, -1].cpu().numpy())
            axes[i, 3].set_title('Original Last Frame')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Main function for testing the model"""
    # Configuration
    config = ModelConfig()
    
    # Create model
    model = SimpleMagvitModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    data_dir = './synthetic_dataset'
    if not os.path.exists(data_dir):
        print("❌ Synthetic dataset not found. Run synthetic_video_generator.py first!")
        return
    
    loaders, datasets = create_data_loaders(data_dir, batch_size=4)
    print(f"📊 Data loaded: {len(datasets['train'])} train, {len(datasets['val'])} val, {len(datasets['test'])} test")
    
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Test forward pass
    sample_batch = next(iter(loaders['train']))
    video = sample_batch['video'].to(device)
    
    print(f"🧪 Testing forward pass with video shape: {video.shape}")
    
    with torch.no_grad():
        result = model(video, 'frame_prediction')
        print(f"✅ Forward pass successful!")
        print(f"   Logits shape: {result['logits'].shape}")
        print(f"   Token IDs shape: {result['token_ids'].shape}")
    
    # Visualize results
    print("🎨 Creating visualization...")
    fig = visualize_results(model, datasets['test'], device, num_samples=2)
    fig.savefig('./model_test_results.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved to model_test_results.png")

if __name__ == "__main__":
    main() 