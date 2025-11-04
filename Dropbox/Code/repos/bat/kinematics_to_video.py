#!/usr/bin/env python3
"""
Convert Kinematics CSV Examples to Video Frames for MAGVIT Training
Creates video frames from trajectory data with augmentation support
"""

import numpy as np
import pandas as pd
import cv2
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class VideoConfig:
    """Configuration for video generation from kinematics"""
    width: int = 64
    height: int = 64
    frames: int = 16
    fps: int = 8
    object_size: int = 5
    trail_length: int = 3  # Number of previous positions to show as trail


class KinematicsVideoGenerator:
    """Generate video frames from kinematics trajectory data"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.setup_colors()
        
    def setup_colors(self):
        """Define color palette for objects and trails"""
        self.object_color = (255, 255, 255)  # White object
        self.trail_colors = [
            (200, 200, 200),  # Light gray (recent)
            (150, 150, 150),  # Medium gray
            (100, 100, 100),  # Dark gray (old)
        ]
        self.background_color = (0, 0, 0)  # Black background
        
    def normalize_trajectory(self, positions: np.ndarray) -> np.ndarray:
        """Normalize trajectory to fit within video frame with padding"""
        # Get bounding box
        min_x, min_y = positions.min(axis=0)
        max_x, max_y = positions.max(axis=0)
        
        # Calculate scale to fit with padding
        padding = 0.1  # 10% padding
        width = max_x - min_x
        height = max_y - min_y
        
        # Handle edge cases
        if width == 0:
            width = 1.0
        if height == 0:
            height = 1.0
            
        # Scale to fit in frame with padding
        scale_x = (self.config.width * (1 - 2 * padding)) / width
        scale_y = (self.config.height * (1 - 2 * padding)) / height
        scale = min(scale_x, scale_y)
        
        # Center the trajectory
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Normalize positions
        normalized = np.zeros_like(positions)
        normalized[:, 0] = (positions[:, 0] - center_x) * scale + self.config.width / 2
        normalized[:, 1] = (positions[:, 1] - center_y) * scale + self.config.height / 2
        
        # Clip to frame bounds
        normalized[:, 0] = np.clip(normalized[:, 0], 0, self.config.width - 1)
        normalized[:, 1] = np.clip(normalized[:, 1], 0, self.config.height - 1)
        
        return normalized
    
    def apply_augmentation(self, positions: np.ndarray, 
                          rotation: Optional[float] = None,
                          scale: Optional[float] = None,
                          noise: Optional[float] = None) -> np.ndarray:
        """Apply augmentation transformations to trajectory"""
        augmented = positions.copy()
        center = np.mean(augmented, axis=0)
        
        # Rotation
        if rotation is not None:
            angle_rad = np.deg2rad(rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            augmented = (augmented - center) @ R.T + center
        
        # Scale
        if scale is not None:
            augmented = (augmented - center) * scale + center
        
        # Noise
        if noise is not None and noise > 0:
            augmented += np.random.normal(0, noise, augmented.shape)
        
        return augmented
    
    def draw_object(self, frame: np.ndarray, position: Tuple[float, float], 
                   trail: Optional[List[Tuple[float, float]]] = None):
        """Draw object and trail on frame"""
        x, y = int(position[0]), int(position[1])
        size = self.config.object_size
        
        # Draw trail (previous positions)
        if trail is not None:
            for i, trail_pos in enumerate(trail[-self.config.trail_length:]):
                if 0 <= trail_pos[0] < self.config.width and 0 <= trail_pos[1] < self.config.height:
                    trail_x, trail_y = int(trail_pos[0]), int(trail_pos[1])
                    color = self.trail_colors[min(i, len(self.trail_colors) - 1)]
                    cv2.circle(frame, (trail_x, trail_y), size // 2, color, -1)
        
        # Draw main object
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            cv2.circle(frame, (x, y), size, self.object_color, -1)
            # Add a small highlight
            cv2.circle(frame, (x - 1, y - 1), size // 3, (255, 255, 255), -1)
    
    def trajectory_to_video(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None,
                           num_frames: Optional[int] = None) -> np.ndarray:
        """Convert trajectory to video frames"""
        if num_frames is None:
            num_frames = self.config.frames
        
        # Normalize trajectory
        normalized_positions = self.normalize_trajectory(positions)
        
        # Sample frames evenly from trajectory
        total_points = len(normalized_positions)
        if total_points < num_frames:
            # Interpolate if we have fewer points than frames
            indices = np.linspace(0, total_points - 1, num_frames).astype(int)
        else:
            # Sample evenly
            indices = np.linspace(0, total_points - 1, num_frames).astype(int)
        
        # Generate frames
        frames = []
        trail = []
        
        for i, idx in enumerate(indices):
            # Create black frame
            frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            
            # Get current position
            current_pos = (normalized_positions[idx, 0], normalized_positions[idx, 1])
            
            # Draw object with trail
            self.draw_object(frame, current_pos, trail)
            
            # Update trail
            trail.append(current_pos)
            if len(trail) > self.config.trail_length:
                trail.pop(0)
            
            frames.append(frame)
        
        # Convert to numpy array: (T, H, W, C)
        video = np.stack(frames, axis=0)
        return video
    
    def create_masked_video(self, video: np.ndarray, mask_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Create masked version of video for prediction task
        
        Args:
            video: Full video (T, H, W, C)
            mask_ratio: Ratio of frames to mask out
            
        Returns:
            masked_video: Video with masked frames (black)
            mask: Boolean array indicating which frames are masked
        """
        num_frames = video.shape[0]
        num_masked = int(num_frames * mask_ratio)
        
        # Randomly select frames to mask
        masked_indices = np.random.choice(num_frames, num_masked, replace=False)
        mask = np.zeros(num_frames, dtype=bool)
        mask[masked_indices] = True
        
        # Create masked video
        masked_video = video.copy()
        masked_video[mask] = 0  # Black out masked frames
        
        return masked_video, mask


class KinematicsDatasetGenerator:
    """Generate video dataset from kinematics CSV files"""
    
    def __init__(self, input_dir: str, output_dir: str, config: VideoConfig):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self.generator = KinematicsVideoGenerator(config)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos').mkdir(exist_ok=True)
        (self.output_dir / 'tasks').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
    
    def generate_dataset(self, num_augmentations: int = 3, split_ratios: Optional[Dict[str, float]] = None):
        """Generate video dataset from kinematics CSV files with augmentation"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.15, 'test': 0.05}
        
        # Find all CSV files
        csv_files = list(self.input_dir.glob('*.csv'))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.input_dir}")
        
        print(f"🎬 Converting {len(csv_files)} kinematics examples to videos...")
        print(f"📊 With {num_augmentations} augmentations per example = {len(csv_files) * (1 + num_augmentations)} total videos")
        print(f"📊 Config: {self.config.width}x{self.config.height}, {self.config.frames} frames")
        
        all_metadata = []
        video_idx = 0
        
        for csv_file in tqdm(csv_files, desc="Processing kinematics"):
            # Load CSV
            df = pd.read_csv(csv_file)
            
            # Extract trajectory data
            positions = df[['x', 'y']].values.astype(np.float32)
            velocities = df[['vx', 'vy']].values.astype(np.float32) if 'vx' in df.columns else None
            
            # Generate base video (no augmentation)
            base_video = self.generator.trajectory_to_video(positions, velocities)
            
            # Create task variants for base video
            self._save_video_and_tasks(f"video_{video_idx:06d}", base_video, csv_file.stem, all_metadata)
            video_idx += 1
            
            # Generate augmented versions
            for aug_idx in range(num_augmentations):
                # Random augmentation parameters
                rotation = random.uniform(-180, 180)
                scale = random.uniform(0.8, 1.2)
                noise = random.uniform(0, 0.5)
                
                # Apply augmentation
                augmented_positions = self.generator.apply_augmentation(
                    positions, rotation=rotation, scale=scale, noise=noise
                )
                
                # Generate video
                aug_video = self.generator.trajectory_to_video(augmented_positions, velocities)
                
                # Create task variants
                video_id = f"video_{video_idx:06d}"
                self._save_video_and_tasks(video_id, aug_video, f"{csv_file.stem}_aug{aug_idx}", all_metadata)
                video_idx += 1
        
        # Create splits
        splits = self._create_splits(all_metadata, split_ratios)
        
        # Save dataset metadata
        dataset_metadata = {
            'config': {
                'width': self.config.width,
                'height': self.config.height,
                'frames': self.config.frames,
                'fps': self.config.fps,
            },
            'num_videos': len(all_metadata),
            'num_original': len(csv_files),
            'num_augmentations': num_augmentations,
            'split_ratios': split_ratios,
            'videos': all_metadata
        }
        
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Create split files
        for split_name, video_ids in splits.items():
            with open(self.output_dir / f'{split_name}_split.txt', 'w') as f:
                f.write('\n'.join(video_ids))
        
        print(f"\n✅ Dataset generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Total videos: {len(all_metadata)}")
        print(f"📊 Splits: {[(k, len(v)) for k, v in splits.items()]}")
        
        return len(all_metadata)
    
    def _save_video_and_tasks(self, video_id: str, video: np.ndarray, 
                             source_name: str, metadata_list: List[Dict]):
        """Save video and create task variants"""
        # Save video
        video_path = self.output_dir / 'videos' / f"{video_id}.npy"
        np.save(video_path, video)
        
        # Create task variants
        tasks = {}
        
        # Task 1: Frame Prediction (predict last half from first half)
        mid_point = self.config.frames // 2
        tasks['frame_prediction'] = {
            'input': video[:mid_point],
            'target': video[mid_point:],
            'task_type': 'frame_prediction'
        }
        
        # Task 2: Masked Frame Prediction (randomly mask frames)
        mask_ratio = random.uniform(0.3, 0.7)
        masked_video, mask = self.generator.create_masked_video(video, mask_ratio)
        tasks['frame_prediction_masked'] = {
            'input': masked_video,
            'target': video,
            'mask': mask,
            'task_type': 'frame_prediction_masked'
        }
        
        # Save task data
        for task_name, task_data in tasks.items():
            np.save(self.output_dir / 'tasks' / f"{video_id}_{task_name}_input.npy", task_data['input'])
            np.save(self.output_dir / 'tasks' / f"{video_id}_{task_name}_target.npy", task_data['target'])
            if 'mask' in task_data:
                np.save(self.output_dir / 'tasks' / f"{video_id}_{task_name}_mask.npy", task_data['mask'])
        
        # Update metadata
        metadata_list.append({
            'video_id': video_id,
            'source': source_name,
            'shape': video.shape,
            'tasks': list(tasks.keys())
        })
    
    def _create_splits(self, metadata: List[Dict], split_ratios: Dict[str, float]) -> Dict[str, List[str]]:
        """Create train/val/test splits"""
        total = len(metadata)
        train_end = int(total * split_ratios['train'])
        val_end = train_end + int(total * split_ratios['val'])
        
        splits = {'train': [], 'val': [], 'test': []}
        for i, meta in enumerate(metadata):
            if i < train_end:
                splits['train'].append(meta['video_id'])
            elif i < val_end:
                splits['val'].append(meta['video_id'])
            else:
                splits['test'].append(meta['video_id'])
        
        return splits


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert kinematics CSV to video dataset for MAGVIT')
    parser.add_argument('--input_dir', type=str, default='./kinematics_examples',
                       help='Input directory with CSV files')
    parser.add_argument('--output_dir', type=str, default='./magvit/kinematics_dataset',
                       help='Output directory for video dataset')
    parser.add_argument('--num_augmentations', type=int, default=3,
                       help='Number of augmentations per example')
    parser.add_argument('--width', type=int, default=64,
                       help='Video width')
    parser.add_argument('--height', type=int, default=64,
                       help='Video height')
    parser.add_argument('--frames', type=int, default=16,
                       help='Number of frames per video')
    
    args = parser.parse_args()
    
    # Create configuration
    config = VideoConfig(
        width=args.width,
        height=args.height,
        frames=args.frames
    )
    
    # Generate dataset
    generator = KinematicsDatasetGenerator(args.input_dir, args.output_dir, config)
    num_videos = generator.generate_dataset(args.num_augmentations)
    
    print(f"\n✅ Generated {num_videos} videos from kinematics examples!")


if __name__ == '__main__':
    main()

