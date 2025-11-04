#!/usr/bin/env python3
"""
Simple MAGVIT Training Example (PyTorch-based)
This uses the working PyTorch implementation from the MAGVIT project
"""

import sys
import os
import subprocess
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Run MAGVIT training example"""
    print("=" * 70)
    print("MAGVIT Training Example (PyTorch-based)")
    print("=" * 70)
    
    # Check if synthetic dataset exists
    data_dir = os.path.join(os.path.dirname(__file__), 'synthetic_dataset')
    
    if not os.path.exists(data_dir):
        print("\n📊 Generating synthetic dataset...")
        print("   This will create a synthetic video dataset for training")
        
        # Generate dataset
        from synthetic_video_generator import SyntheticDatasetGenerator, VideoConfig
        
        config = VideoConfig(
            width=64,
            height=64,
            frames=16,
            num_objects=2
        )
        
        dataset_generator = SyntheticDatasetGenerator(data_dir, config)
        dataset_generator.generate_dataset(num_videos=100, split_ratios={'train': 0.8, 'val': 0.15, 'test': 0.05})
        
        print("✅ Synthetic dataset generated!")
    else:
        print(f"\n✅ Using existing dataset at: {data_dir}")
    
    # Run training
    print("\n🚀 Starting MAGVIT training...")
    print("   This will train the model on synthetic video data")
    
    # Import and run training
    from train_simple_magvit import main as train_main
    import argparse
    
    # Create minimal args for training
    args = argparse.Namespace(
        data_dir=data_dir,
        output_dir=os.path.join(os.path.dirname(__file__), 'experiments', 'simple_magvit'),
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        epochs=5,  # Reduced for quick example
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        save_freq=2,
        vis_freq=2,
        use_wandb=False,
        run_name='example',
        num_workers=2,
        seed=42
    )
    
    # Set args in sys.argv for argparse
    sys.argv = ['train_simple_magvit.py']
    
    try:
        # Create trainer and run
        from train_simple_magvit import VideoTrainer, ModelConfig
        
        config = ModelConfig(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        )
        
        trainer = VideoTrainer(config, args)
        trainer.train()
        
        print("\n✅ MAGVIT training completed successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        
        return 0
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
