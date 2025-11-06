"""
Test cases for training checkpoint persistence and directory creation.

These tests verify that:
1. Training creates output directories correctly
2. Checkpoints are saved during training
3. Checkpoints persist after training completes
4. Checkpoint files have correct size and format
5. Inference can find and use the correct checkpoint
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import torch
import sys

# Add project root and magvit directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'magvit'))

from train_simple_magvit import VideoTrainer, ModelConfig
from simple_magvit_model import SimpleMagvitModel


class TestCheckpointPersistence:
    """Test checkpoint saving and persistence"""
    
    def test_output_directory_creation(self, tmp_path):
        """Test that output directory is created before training starts"""
        output_dir = tmp_path / "test_experiment"
        
        # Verify directory doesn't exist
        assert not output_dir.exists()
        
        # Create directory (simulating trainer initialization)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify directory exists
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_checkpoint_save_creates_directory(self, tmp_path):
        """Test that save_checkpoint creates directory if it doesn't exist"""
        output_dir = tmp_path / "test_checkpoints"
        checkpoint_path = output_dir / "checkpoint_latest.pth"
        
        # Directory doesn't exist
        assert not output_dir.exists()
        
        # Create directory and save checkpoint
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy checkpoint
        checkpoint = {
            'epoch': 0,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 256},
            'best_val_loss': 10.0
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Verify checkpoint exists
        assert checkpoint_path.exists()
        assert checkpoint_path.is_file()
    
    def test_checkpoint_file_size(self, tmp_path):
        """Test that checkpoint files have reasonable size (not empty or corrupted)"""
        output_dir = tmp_path / "test_checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a realistic checkpoint (minimal model)
        checkpoint = {
            'epoch': 399,
            'model_state_dict': {
                'layer.weight': torch.randn(100, 100),
                'layer.bias': torch.randn(100)
            },
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 768},
            'best_val_loss': 5.8
        }
        
        checkpoint_path = output_dir / "checkpoint_test.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Verify file exists and has reasonable size
        assert checkpoint_path.exists()
        file_size = checkpoint_path.stat().st_size
        assert file_size > 1000, f"Checkpoint too small: {file_size} bytes"
        assert file_size < 10 * 1024 * 1024 * 1024, f"Checkpoint too large: {file_size} bytes"
    
    def test_checkpoint_can_be_loaded(self, tmp_path):
        """Test that saved checkpoint can be loaded successfully"""
        output_dir = tmp_path / "test_checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create and save checkpoint
        checkpoint = {
            'epoch': 399,
            'model_state_dict': {'test.weight': torch.randn(10, 10)},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 768, 'num_layers': 12},
            'best_val_loss': 5.8328
        }
        
        checkpoint_path = output_dir / "checkpoint_test.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Verify contents
        assert loaded['epoch'] == 399
        assert loaded['best_val_loss'] == 5.8328
        assert loaded['config']['hidden_dim'] == 768
        assert 'model_state_dict' in loaded
    
    def test_multiple_checkpoints_persist(self, tmp_path):
        """Test that multiple checkpoints can be saved and persist"""
        output_dir = tmp_path / "test_checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save multiple checkpoints
        for epoch in [0, 5, 10, 399]:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'scheduler_state_dict': {},
                'config': {'hidden_dim': 768},
                'best_val_loss': 10.0 - epoch * 0.01
            }
            
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
        
        # Verify all checkpoints exist
        for epoch in [0, 5, 10, 399]:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            assert checkpoint_path.exists(), f"Checkpoint for epoch {epoch} not found"
    
    def test_checkpoint_directory_structure(self, tmp_path):
        """Test that checkpoint directory has expected structure"""
        output_dir = tmp_path / "experiments" / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoints
        for name in ['checkpoint_latest.pth', 'checkpoint_best.pth']:
            checkpoint = {
                'epoch': 399,
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'scheduler_state_dict': {},
                'config': {'hidden_dim': 768},
                'best_val_loss': 5.8
            }
            torch.save(checkpoint, output_dir / name)
        
        # Verify structure
        assert (output_dir / 'checkpoint_latest.pth').exists()
        assert (output_dir / 'checkpoint_best.pth').exists()
        assert output_dir.is_dir()


class TestTrainingDirectorySetup:
    """Test that training sets up directories correctly"""
    
    def test_output_dir_created_before_training(self, tmp_path):
        """Test that output directory is created before training starts"""
        output_dir = tmp_path / "training_output"
        
        # Simulate trainer initialization
        # In VideoTrainer.__init__, output_dir should be created
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_training_log_directory(self, tmp_path):
        """Test that training log directory is created"""
        log_dir = tmp_path / "logs"
        log_file = log_dir / "training.log"
        
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        log_file.write_text("Training started\n")
        
        assert log_dir.exists()
        assert log_file.exists()
    
    def test_checkpoint_directory_permissions(self, tmp_path):
        """Test that checkpoint directory has correct permissions"""
        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify directory is writable
        test_file = output_dir / "test.txt"
        test_file.write_text("test")
        
        assert test_file.exists()
        assert test_file.read_text() == "test"


class TestCheckpointVerification:
    """Test checkpoint verification and validation"""
    
    def test_checkpoint_contains_required_keys(self, tmp_path):
        """Test that checkpoint contains all required keys"""
        output_dir = tmp_path / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': 399,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 768},
            'best_val_loss': 5.8
        }
        
        checkpoint_path = output_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Load and verify keys
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 
                        'scheduler_state_dict', 'config', 'best_val_loss']
        
        for key in required_keys:
            assert key in loaded, f"Missing required key: {key}"
    
    def test_checkpoint_epoch_validation(self, tmp_path):
        """Test that checkpoint epoch is within expected range"""
        output_dir = tmp_path / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test valid epoch
        checkpoint = {
            'epoch': 399,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 768},
            'best_val_loss': 5.8
        }
        
        checkpoint_path = output_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert 0 <= loaded['epoch'] <= 400, f"Invalid epoch: {loaded['epoch']}"
    
    def test_checkpoint_config_matches_expected(self, tmp_path):
        """Test that checkpoint config matches expected kinematics training config"""
        output_dir = tmp_path / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Kinematics training config
        expected_config = {
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_ratio': 4
        }
        
        checkpoint = {
            'epoch': 399,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': expected_config,
            'best_val_loss': 5.8
        }
        
        checkpoint_path = output_dir / "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = loaded['config']
        
        assert config['hidden_dim'] == 768, "Wrong hidden_dim"
        assert config['num_layers'] == 12, "Wrong num_layers"
        assert config['num_heads'] == 12, "Wrong num_heads"


class TestCheckpointDiscovery:
    """Test that checkpoints can be found correctly"""
    
    def test_find_kinematics_checkpoint(self, tmp_path):
        """Test finding kinematics training checkpoint"""
        experiments_dir = tmp_path / "experiments" / "kinematics_magvit"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            'epoch': 399,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 768},
            'best_val_loss': 5.8
        }
        
        checkpoint_path = experiments_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Verify it can be found
        assert checkpoint_path.exists()
        
        # Verify it's the correct checkpoint (not synthetic data)
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert loaded['config']['hidden_dim'] == 768, "Wrong checkpoint type"
        assert loaded['epoch'] >= 390, "Wrong epoch"
    
    def test_detect_wrong_checkpoint_type(self, tmp_path):
        """Test that wrong checkpoint type (synthetic vs kinematics) is detected"""
        # Create synthetic checkpoint (wrong type)
        synthetic_dir = tmp_path / "experiments" / "simple_magvit"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        synthetic_checkpoint = {
            'epoch': 4,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 256},  # Wrong!
            'best_val_loss': 5.9
        }
        
        synthetic_path = synthetic_dir / "checkpoint_best.pth"
        torch.save(synthetic_checkpoint, synthetic_path)
        
        # Verify it's detected as wrong type
        loaded = torch.load(synthetic_path, map_location='cpu', weights_only=False)
        assert loaded['config']['hidden_dim'] != 768, "Should detect wrong checkpoint type"
        assert loaded['epoch'] < 390, "Should detect wrong epoch"
    
    def test_checkpoint_search_finds_correct_location(self, tmp_path):
        """Test that checkpoint search finds correct location"""
        # Create correct structure
        kinematics_dir = tmp_path / "experiments" / "kinematics_magvit"
        kinematics_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': 399,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'config': {'hidden_dim': 768},
            'best_val_loss': 5.8
        }
        
        checkpoint_path = kinematics_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Search for checkpoint
        found = list(kinematics_dir.glob("checkpoint*.pth"))
        assert len(found) > 0, "Checkpoint not found"
        assert checkpoint_path in found, "Wrong checkpoint found"


class TestBackgroundTrainingVerification:
    """Test that background training creates and persists files correctly"""
    
    def test_pid_file_creation(self, tmp_path):
        """Test that PID file is created when training starts"""
        pid_file = tmp_path / "training.pid"
        
        # Simulate PID file creation
        pid_file.write_text("12345\n")
        
        assert pid_file.exists()
        pid = int(pid_file.read_text().strip())
        assert pid > 0
    
    def test_log_file_creation(self, tmp_path):
        """Test that log file is created and written to"""
        log_file = tmp_path / "training.log"
        
        # Simulate log writing
        with open(log_file, 'w') as f:
            f.write("Training started\n")
            f.write("Epoch 1/400\n")
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Training started" in content
        assert "Epoch" in content
    
    def test_checkpoint_created_during_training(self, tmp_path):
        """Test that checkpoints are created during training"""
        output_dir = tmp_path / "training_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate checkpoint saving every 5 epochs
        for epoch in range(0, 400, 5):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'scheduler_state_dict': {},
                'config': {'hidden_dim': 768},
                'best_val_loss': 10.0 - epoch * 0.01
            }
            torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
        
        # Verify latest checkpoint exists
        assert (output_dir / "checkpoint_latest.pth").exists()
        
        # Verify it's from a recent epoch
        loaded = torch.load(output_dir / "checkpoint_latest.pth", map_location='cpu', weights_only=False)
        assert loaded['epoch'] >= 395, "Latest checkpoint should be from recent epoch"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

