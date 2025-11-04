#!/usr/bin/env python3
"""
Synthetic Video Data Generator for MAGVIT-Style Foundation Model
Creates simple geometric shapes with basic physics for multi-task video learning.

Tasks supported:
- Frame Prediction: Predict next N frames
- Frame Interpolation: Generate intermediate frames
- Video Inpainting: Fill masked regions
- Video Outpainting: Extend video boundaries
- Conditional Generation: Generate based on shape/motion parameters
"""

import numpy as np
import cv2
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class VideoConfig:
    """Configuration for synthetic video generation"""
    width: int = 64
    height: int = 64
    frames: int = 16
    fps: int = 8
    num_objects: int = 2
    object_types: Optional[List[str]] = None
    motion_types: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.object_types is None:
            self.object_types = ['circle', 'rectangle', 'triangle']
        if self.motion_types is None:
            self.motion_types = ['linear', 'circular', 'bounce', 'stationary']

@dataclass
class PhysicsObject:
    """Simple physics object for synthetic videos"""
    x: float
    y: float
    vx: float
    vy: float
    size: float
    color: Tuple[int, int, int]
    shape: str
    motion_type: str
    
class SyntheticVideoGenerator:
    """Generate synthetic videos with simple physics and geometric shapes"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.setup_colors()
        
    def setup_colors(self):
        """Define color palette for objects"""
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
    
    def create_random_object(self) -> PhysicsObject:
        """Create a random physics object"""
        x = random.uniform(0.2, 0.8) * self.config.width
        y = random.uniform(0.2, 0.8) * self.config.height
        
        # Random velocity
        vx = random.uniform(-2, 2)
        vy = random.uniform(-2, 2)
        
        # Random size
        size = random.uniform(5, 15)
        
        # Random properties
        color = random.choice(self.colors)
        assert self.config.object_types is not None
        assert self.config.motion_types is not None
        shape = random.choice(self.config.object_types)
        motion_type = random.choice(self.config.motion_types)
        
        return PhysicsObject(x, y, vx, vy, size, color, shape, motion_type)
    
    def update_object_physics(self, obj: PhysicsObject, frame_idx: int):
        """Update object position based on physics"""
        if obj.motion_type == 'linear':
            obj.x += obj.vx
            obj.y += obj.vy
            
        elif obj.motion_type == 'circular':
            # Circular motion around center
            center_x = self.config.width / 2
            center_y = self.config.height / 2
            radius = 20
            angle = (frame_idx * 0.2) + hash(str(obj.color)) % 100
            obj.x = center_x + radius * np.cos(angle)
            obj.y = center_y + radius * np.sin(angle)
            
        elif obj.motion_type == 'bounce':
            obj.x += obj.vx
            obj.y += obj.vy
            
            # Bounce off walls
            if obj.x <= obj.size or obj.x >= self.config.width - obj.size:
                obj.vx *= -1
            if obj.y <= obj.size or obj.y >= self.config.height - obj.size:
                obj.vy *= -1
                
        # elif obj.motion_type == 'stationary': do nothing
        
        # Keep objects in bounds
        obj.x = np.clip(obj.x, obj.size, self.config.width - obj.size)
        obj.y = np.clip(obj.y, obj.size, self.config.height - obj.size)
    
    def draw_object(self, frame: np.ndarray, obj: PhysicsObject):
        """Draw object on frame"""
        x, y = int(obj.x), int(obj.y)
        size = int(obj.size)
        
        if obj.shape == 'circle':
            cv2.circle(frame, (x, y), size, obj.color, -1)
            
        elif obj.shape == 'rectangle':
            cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), obj.color, -1)
            
        elif obj.shape == 'triangle':
            pts = np.array([
                [x, y-size],
                [x-size, y+size],
                [x+size, y+size]
            ], np.int32)
            cv2.fillPoly(frame, [pts], obj.color)
    
    def generate_video(self, video_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a single synthetic video"""
        # Initialize objects
        objects = [self.create_random_object() for _ in range(self.config.num_objects)]
        
        # Generate frames
        frames = []
        object_trajectories = []
        
        for frame_idx in range(self.config.frames):
            # Create black frame
            frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            
            # Update and draw objects
            frame_objects = []
            for obj in objects:
                self.update_object_physics(obj, frame_idx)
                self.draw_object(frame, obj)
                
                # Store object state
                frame_objects.append({
                    'x': obj.x, 'y': obj.y, 'vx': obj.vx, 'vy': obj.vy,
                    'size': obj.size, 'color': obj.color, 'shape': obj.shape,
                    'motion_type': obj.motion_type
                })
            
            frames.append(frame)
            object_trajectories.append(frame_objects)
        
        # Convert to numpy array
        video = np.stack(frames, axis=0)  # Shape: (T, H, W, C)
        
        # Create metadata
        metadata = {
            'video_id': video_id,
            'config': self.config.__dict__,
            'objects': object_trajectories,
            'shape': video.shape
        }
        
        return video, metadata
    
    def create_task_variants(self, video: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create different task variants from base video"""
        tasks = {}
        
        # Task 1: Frame Prediction (predict last 8 frames from first 8)
        mid_point = self.config.frames // 2
        tasks['frame_prediction'] = {
            'input': video[:mid_point],      # First half
            'target': video[mid_point:],     # Second half
            'task_type': 'frame_prediction'
        }
        
        # Task 2: Frame Interpolation (predict middle frames from first+last)
        if self.config.frames >= 4:
            boundary_frames = 2
            tasks['frame_interpolation'] = {
                'input': np.concatenate([video[:boundary_frames], video[-boundary_frames:]], axis=0),
                'target': video[boundary_frames:-boundary_frames],
                'task_type': 'frame_interpolation'
            }
        
        # Task 3: Video Inpainting (mask central region)
        masked_video = video.copy()
        h, w = video.shape[1], video.shape[2]
        mask_h, mask_w = h//3, w//3
        start_h, start_w = h//3, w//3
        
        # Create mask
        mask = np.ones((h, w), dtype=bool)
        mask[start_h:start_h+mask_h, start_w:start_w+mask_w] = False
        
        # Apply mask to video
        for t in range(len(masked_video)):
            masked_video[t][~mask] = 0  # Black out masked region
        
        tasks['video_inpainting'] = {
            'input': masked_video,
            'target': video,
            'mask': mask,
            'task_type': 'video_inpainting'
        }
        
        # Task 4: Video Outpainting (predict full video from center crop)
        crop_size = min(h, w) // 2
        start_h, start_w = (h - crop_size) // 2, (w - crop_size) // 2
        cropped_video = video[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
        
        tasks['video_outpainting'] = {
            'input': cropped_video,
            'target': video,
            'task_type': 'video_outpainting'
        }
        
        return tasks

class SyntheticDatasetGenerator:
    """Generate complete synthetic dataset for training"""
    
    def __init__(self, output_dir: str, config: VideoConfig):
        self.output_dir = Path(output_dir)
        self.config = config
        self.generator = SyntheticVideoGenerator(config)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos').mkdir(exist_ok=True)
        (self.output_dir / 'tasks').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
    
    def generate_dataset(self, num_videos: int = 1000, split_ratios: Optional[Dict[str, float]] = None):
        """Generate complete dataset"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.15, 'test': 0.05}
        
        print(f"🎬 Generating {num_videos} synthetic videos...")
        print(f"📊 Config: {self.config.width}x{self.config.height}, {self.config.frames} frames")
        
        all_metadata = []
        
        for i in tqdm(range(num_videos), desc="Generating videos"):
            video_id = f"video_{i:06d}"
            
            # Generate base video
            video, metadata = self.generator.generate_video(video_id)
            
            # Create task variants
            tasks = self.generator.create_task_variants(video, metadata)
            
            # Determine split
            split = self.determine_split(i, num_videos, split_ratios)
            
            # Save video
            video_path = self.output_dir / 'videos' / f"{video_id}.npy"
            np.save(video_path, video)
            
            # Save tasks
            tasks_path = self.output_dir / 'tasks' / f"{video_id}.json"
            # Convert numpy arrays to lists for JSON serialization
            tasks_json = {}
            for task_name, task_data in tasks.items():
                tasks_json[task_name] = {
                    'input_shape': task_data['input'].shape,
                    'target_shape': task_data['target'].shape,
                    'task_type': task_data['task_type']
                }
                # Save numpy arrays separately
                np.save(self.output_dir / 'tasks' / f"{video_id}_{task_name}_input.npy", task_data['input'])
                np.save(self.output_dir / 'tasks' / f"{video_id}_{task_name}_target.npy", task_data['target'])
                if 'mask' in task_data:
                    np.save(self.output_dir / 'tasks' / f"{video_id}_{task_name}_mask.npy", task_data['mask'])
            
            with open(tasks_path, 'w') as f:
                json.dump(tasks_json, f, indent=2)
            
            # Update metadata
            metadata['split'] = split
            metadata['tasks'] = list(tasks.keys())
            all_metadata.append(metadata)
        
        # Save dataset metadata
        dataset_metadata = {
            'config': self.config.__dict__,
            'num_videos': num_videos,
            'split_ratios': split_ratios,
            'videos': all_metadata
        }
        
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Create split files
        splits = {'train': [], 'val': [], 'test': []}
        for metadata in all_metadata:
            splits[metadata['split']].append(metadata['video_id'])
        
        for split_name, video_ids in splits.items():
            with open(self.output_dir / f'{split_name}_split.txt', 'w') as f:
                f.write('\n'.join(video_ids))
        
        print(f"✅ Dataset generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Splits: {[(k, len(v)) for k, v in splits.items()]}")
    
    def determine_split(self, index: int, total: int, ratios: Dict[str, float]) -> str:
        """Determine which split a video belongs to"""
        train_end = int(total * ratios['train'])
        val_end = train_end + int(total * ratios['val'])
        
        if index < train_end:
            return 'train'
        elif index < val_end:
            return 'val'
        else:
            return 'test'
    
    def visualize_sample(self, video_id: Optional[str] = None, save_path: Optional[str] = None):
        """Visualize a sample video and its tasks"""
        if video_id is None:
            # Get random video
            video_files = list((self.output_dir / 'videos').glob('*.npy'))
            if not video_files:
                print("No videos found in dataset!")
                return
            video_id = video_files[0].stem
        
        # Load video
        video = np.load(self.output_dir / 'videos' / f"{video_id}.npy")
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Show original video frames
        for i in range(4):
            if i < len(video):
                axes[0, i].imshow(video[i])
                axes[0, i].set_title(f'Frame {i}')
                axes[0, i].axis('off')
        
        # Show task examples
        task_names = ['frame_prediction', 'frame_interpolation', 'video_inpainting', 'video_outpainting']
        
        for i, task_name in enumerate(task_names):
            input_path = self.output_dir / 'tasks' / f"{video_id}_{task_name}_input.npy"
            if input_path.exists():
                task_input = np.load(input_path)
                if len(task_input) > 0:
                    axes[1, i].imshow(task_input[0])
                    axes[1, i].set_title(f'{task_name}\n(input)')
                    axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

def main():
    """Main function to generate synthetic dataset"""
    parser = argparse.ArgumentParser(description='Generate synthetic video dataset for MAGVIT training')
    parser.add_argument('--output_dir', type=str, default='./synthetic_dataset', 
                        help='Output directory for dataset')
    parser.add_argument('--num_videos', type=int, default=1000,
                        help='Number of videos to generate')
    parser.add_argument('--width', type=int, default=64,
                        help='Video width')
    parser.add_argument('--height', type=int, default=64,
                        help='Video height')
    parser.add_argument('--frames', type=int, default=16,
                        help='Number of frames per video')
    parser.add_argument('--num_objects', type=int, default=2,
                        help='Number of objects per video')
    parser.add_argument('--visualize', action='store_true',
                        help='Create sample visualization')
    
    args = parser.parse_args()
    
    # Create configuration
    config = VideoConfig(
        width=args.width,
        height=args.height,
        frames=args.frames,
        num_objects=args.num_objects
    )
    
    # Generate dataset
    dataset_generator = SyntheticDatasetGenerator(args.output_dir, config)
    dataset_generator.generate_dataset(args.num_videos)
    
    # Create visualization if requested
    if args.visualize:
        vis_path = os.path.join(args.output_dir, 'sample_visualization.png')
        dataset_generator.visualize_sample(save_path=vis_path)

if __name__ == "__main__":
    main() 