#!/usr/bin/env python3
"""
Cyber Evan Project - Main MacBook Interface
Connects to vast.ai instance and manages project execution

Usage:
    python3 main_macbook.py [--config config.yaml] [--setup] [--test] [--install-deps]
    
Note: Always use 'python3' (not 'python') on macOS to avoid Python 2.7 compatibility issues.
"""

import yaml
import subprocess
import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional


class VastAIConnector:
    """Manages connection and operations with vast.ai instance"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration file"""
        self.config_path = config_path
        self.config = self._load_config()
        ssh_key_path = self.config['ssh'].get('key_path')
        self.ssh_key = os.path.expanduser(ssh_key_path) if ssh_key_path else None
        self.host = self.config['vast_ai']['public_ip']
        self.user = self.config['vast_ai']['ssh_user']
        self.remote_path = self.config['project']['remote_path']
        self.ssh_port = self.config['ssh'].get('port', 22)
        self.use_password_auth = self.config['ssh'].get('use_password_auth', False)
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"❌ Configuration file {self.config_path} not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _run_ssh_command(self, command: str, capture_output: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Execute SSH command on vast.ai instance"""
        ssh_cmd = ['ssh', '-p', str(self.ssh_port), '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=30']
        
        # Add SSH key if provided
        if self.ssh_key:
            ssh_cmd.extend(['-i', self.ssh_key])
        
        ssh_cmd.extend([f'{self.user}@{self.host}', command])
        
        print(f"🔗 Executing: {' '.join(ssh_cmd[:3])} ... {command}")
        
        try:
            # Use timeout parameter if provided, otherwise use default
            cmd_timeout = timeout if timeout is not None else (None if 'nohup' in command else 60)
            result = subprocess.run(
                ssh_cmd,
                capture_output=capture_output,
                text=True,
                timeout=cmd_timeout
            )
            return result
        except subprocess.TimeoutExpired:
            print(f"❌ SSH command timed out: {command}")
            return subprocess.CompletedProcess(ssh_cmd, 1, "", "Timeout")
        except Exception as e:
            print(f"❌ SSH command failed: {e}")
            return subprocess.CompletedProcess(ssh_cmd, 1, "", str(e))
    
    def test_connection(self) -> bool:
        """Test connection to vast.ai instance"""
        print("🧪 Testing connection to vast.ai instance...")
        print(f"   Host: {self.host}")
        print(f"   User: {self.user}")
        if self.ssh_key:
            print(f"   SSH Key: {self.ssh_key}")
        else:
            print("   SSH Key: None (password authentication)")
        
        # Check if SSH key exists (only if key is specified)
        if self.ssh_key and not os.path.exists(self.ssh_key):
            print(f"❌ SSH key not found at: {self.ssh_key}")
            print("   Please update the key_path in config.yaml")
            return False
        
        # Test SSH connection
        result = self._run_ssh_command(self.config['connection']['test_command'])
        
        if result.returncode == 0:
            print("✅ Connection successful!")
            print(f"   Response: {result.stdout.strip()}")
            return True
        else:
            print("❌ Connection failed!")
            print(f"   Error: {result.stderr.strip()}")
            return False
    
    def setup_environment(self) -> bool:
        """Set up basic environment on vast.ai instance"""
        print("🔧 Setting up environment on vast.ai instance...")
        
        setup_commands = self.config['connection']['setup_commands']
        
        for cmd in setup_commands:
            print(f"   Running: {cmd}")
            result = self._run_ssh_command(cmd)
            
            if result.returncode != 0:
                print(f"❌ Setup command failed: {cmd}")
                print(f"   Error: {result.stderr.strip()}")
                return False
        
        print("✅ Environment setup completed!")
        return True
    
    def install_python_dependencies(self) -> bool:
        """Install Python dependencies on vast.ai instance"""
        print("📦 Installing Python dependencies...")
        
        deps = self.config['python_dependencies']
        deps_str = ' '.join(deps)
        
        # Install Python packages
        install_cmd = f"pip3 install {deps_str} --user"
        result = self._run_ssh_command(install_cmd)
        
        if result.returncode == 0:
            print("✅ Python dependencies installed successfully!")
            return True
        else:
            print("❌ Failed to install Python dependencies!")
            print(f"   Error: {result.stderr.strip()}")
            return False
    
    def install_system_dependencies(self) -> bool:
        """Install system dependencies on vast.ai instance"""
        print("🔧 Installing system dependencies...")
        
        deps = self.config['system_dependencies']
        deps_str = ' '.join(deps)
        
        # Install system packages
        install_cmd = f"apt update && apt install -y {deps_str}"
        result = self._run_ssh_command(install_cmd)
        
        if result.returncode == 0:
            print("✅ System dependencies installed successfully!")
            return True
        else:
            print("❌ Failed to install system dependencies!")
            print(f"   Error: {result.stderr.strip()}")
            return False
    
    def sync_project_files(self) -> bool:
        """Sync project files to vast.ai instance"""
        print("📁 Syncing project files to vast.ai instance...")
        
        local_path = self.config['project']['local_path']
        
        # Create remote directory
        mkdir_cmd = f"mkdir -p {self.remote_path}"
        result = self._run_ssh_command(mkdir_cmd)
        
        if result.returncode != 0:
            print(f"❌ Failed to create remote directory: {result.stderr.strip()}")
            return False
        
        # Sync files using rsync
        rsync_cmd = [
            'rsync',
            '-avz',
            '--delete',
            '-e', f'ssh -p {self.ssh_port} -o StrictHostKeyChecking=no' + (f' -i {self.ssh_key}' if self.ssh_key else ''),
            f'{local_path}/',
            f'{self.user}@{self.host}:{self.remote_path}/'
        ]
        
        print(f"   Syncing: {local_path} -> {self.host}:{self.remote_path}")
        
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Project files synced successfully!")
                return True
            else:
                print("❌ Failed to sync project files!")
                print(f"   Error: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ File sync timed out!")
            return False
    
    def execute_remote_command(self, command: str) -> bool:
        """Execute a command on the vast.ai instance"""
        print(f"🚀 Executing remote command: {command}")
        
        result = self._run_ssh_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Command executed successfully!")
            return True
        else:
            print("❌ Command execution failed!")
            return False
    
    def get_remote_shell(self):
        """Open interactive shell on vast.ai instance"""
        print("🐚 Opening remote shell...")
        print("   Type 'exit' to return to local shell")
        
        ssh_cmd = ['ssh', '-p', str(self.ssh_port), '-o', 'StrictHostKeyChecking=no']
        
        # Add SSH key if provided
        if self.ssh_key:
            ssh_cmd.extend(['-i', self.ssh_key])
        
        ssh_cmd.append(f'{self.user}@{self.host}')
        
        try:
            subprocess.run(ssh_cmd)
        except KeyboardInterrupt:
            print("\n👋 Remote shell closed")
    
    def run_kinematics_examples(self) -> bool:
        """Run kinematics examples generation on vast.ai instance"""
        print("🎯 Running kinematics examples generation...")
        
        command = f"cd {self.remote_path} && python3 generate_kinematics_examples.py"
        result = self._run_ssh_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Kinematics examples generated successfully!")
            
            # List output directory to show what was created
            list_cmd = f"cd {self.remote_path} && ls -la output/ 2>/dev/null || echo 'Output directory not found'"
            self._run_ssh_command(list_cmd)
            return True
        else:
            print("❌ Failed to generate kinematics examples!")
            return False
    
    def download_output_files(self, remote_dir: str = "output", local_dir: str = "output") -> bool:
        """Download output files from vast.ai instance to MacBook"""
        print(f"📥 Downloading output files from vast.ai instance...")
        
        local_path = self.config['project']['local_path']
        remote_output_path = f"{self.remote_path}/{remote_dir}"
        local_output_path = os.path.join(local_path, local_dir)
        
        # Create local output directory if it doesn't exist
        os.makedirs(local_output_path, exist_ok=True)
        
        # Use rsync to download files (reversed direction from sync)
        rsync_cmd = [
            'rsync',
            '-avz',
            '--progress',
            '-e', f'ssh -p {self.ssh_port} -o StrictHostKeyChecking=no' + (f' -i {self.ssh_key}' if self.ssh_key else ''),
            f'{self.user}@{self.host}:{remote_output_path}/',
            f'{local_output_path}/'
        ]
        
        print(f"   Downloading: {self.host}:{remote_output_path} -> {local_output_path}")
        
        try:
            result = subprocess.run(rsync_cmd, capture_output=False, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ Output files downloaded successfully to {local_output_path}/")
                return True
            else:
                print("❌ Failed to download output files!")
                return False
        except subprocess.TimeoutExpired:
            print("❌ File download timed out!")
            return False
        except Exception as e:
            print(f"❌ Error downloading files: {e}")
            return False
    
    def run_magvit_example(self) -> bool:
        """Run MAGVIT simple example on vast.ai instance"""
        print("🎬 Running MAGVIT simple example...")
        
        magvit_config = self.config.get('magvit', {})
        example_script = magvit_config.get('example_script', 'magvit/simple_example.py')
        
        # Change to project directory and run the example
        command = f"cd {self.remote_path} && python3 {example_script}"
        
        result = self._run_ssh_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ MAGVIT example completed successfully!")
            
            # List output directory if it exists
            list_cmd = f"cd {self.remote_path} && ls -la magvit_test_* 2>/dev/null | head -5 || echo 'No test output directories found'"
            self._run_ssh_command(list_cmd)
            return True
        else:
            print("❌ Failed to run MAGVIT example!")
            return False
    
    def run_magvit_training(self) -> bool:
        """Run MAGVIT training example (PyTorch-based) on vast.ai instance"""
        print("🎬 Running MAGVIT training example (PyTorch-based)...")
        print("   This will train on synthetic video data")
        print("   This will take several minutes...")
        
        # Change to project directory and run the training example
        command = f"cd {self.remote_path}/magvit && python3 train_example.py"
        
        result = self._run_ssh_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ MAGVIT training completed successfully!")
            
            # List output directory if it exists
            list_cmd = f"cd {self.remote_path}/magvit && ls -la experiments/simple_magvit/ 2>/dev/null | head -10 || echo 'No training output directories found'"
            self._run_ssh_command(list_cmd)
            return True
        else:
            print("❌ Failed to run MAGVIT training!")
            return False
    
    def convert_kinematics_to_video(self) -> bool:
        """Convert kinematics CSV examples to video dataset for MAGVIT training"""
        print("🎬 Converting kinematics examples to video dataset...")
        print("   This will create video frames from trajectory data with augmentation")
        
        # Get config values
        kinematics_dir = os.path.join(self.remote_path, 'kinematics_examples')
        output_dir = os.path.join(self.remote_path, self.config.get('magvit', {}).get('kinematics_dataset_dir', 'magvit/kinematics_dataset'))
        num_augmentations = self.config.get('magvit', {}).get('num_augmentations', 3)
        
        # Run conversion script
        command = f"cd {self.remote_path} && python3 kinematics_to_video.py --input_dir {kinematics_dir} --output_dir {output_dir} --num_augmentations {num_augmentations}"
        
        result = self._run_ssh_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Kinematics to video conversion completed!")
            
            # Show dataset stats
            stats_cmd = f"cd {self.remote_path} && python3 -c \"import json; f=open('{output_dir}/dataset_metadata.json'); d=json.load(f); print(f'Total videos: {{d[\\'num_videos\\']}}'); print(f'Original examples: {{d[\\'num_original\\']}}'); print(f'Augmentations per example: {{d[\\'num_augmentations\\']}}')\" 2>&1"
            self._run_ssh_command(stats_cmd)
            return True
        else:
            print("❌ Failed to convert kinematics to video!")
            return False
    
    def run_magvit_kinematics_training(self) -> bool:
        """Run MAGVIT training on kinematics-based video dataset"""
        print("🎬 Running MAGVIT training on kinematics dataset...")
        print("   This will train the model to predict missing observations")
        print("   This will take several minutes...")
        
        # Get dataset directory
        dataset_dir = os.path.join(self.remote_path, self.config.get('magvit', {}).get('kinematics_dataset_dir', 'magvit/kinematics_dataset'))
        
        # Check if dataset exists
        check_cmd = f"cd {self.remote_path} && test -d {dataset_dir}/videos && echo 'Dataset exists' || echo 'Dataset not found'"
        check_result = self._run_ssh_command(check_cmd, capture_output=True)
        stdout_text = check_result.stdout.decode('utf-8') if isinstance(check_result.stdout, bytes) else str(check_result.stdout)
        
        if 'Dataset not found' in stdout_text:
            print("⚠️  Dataset not found. Running conversion first...")
            if not self.convert_kinematics_to_video():
                return False
        
        # Run training with kinematics dataset in background using nohup
        # This ensures training continues even if SSH connection drops
        output_dir = os.path.join(self.remote_path, 'magvit/experiments/kinematics_magvit')
        log_file = os.path.join(output_dir, 'training.log')
        pid_file = os.path.join(output_dir, 'training.pid')
        
        # Create output directory first
        mkdir_cmd = f"mkdir -p {output_dir}"
        mkdir_result = self._run_ssh_command(mkdir_cmd, capture_output=True)
        if mkdir_result.returncode != 0:
            print("❌ Failed to create output directory!")
            return False
        
        # Create a Python script to run training (more reliable than inline -c)
        training_script = f"""import sys
import os
sys.path.insert(0, '{self.remote_path}')

from magvit.train_simple_magvit import VideoTrainer, ModelConfig
import argparse

args = argparse.Namespace(
    data_dir='{dataset_dir}',
    output_dir='{output_dir}',
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    epochs=400,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=4.5e-2,
    grad_clip=1.0,
    save_freq=5,
    vis_freq=5,
    use_wandb=False,
    run_name='kinematics',
    num_workers=2,
    seed=42,
    resume_from=None
)

config = ModelConfig(hidden_dim=768, num_layers=12, num_heads=12, mlp_ratio=4)
trainer = VideoTrainer(config, args)
trainer.train(resume_from=args.resume_from)
"""
        
        # Write script to temporary file
        script_path = os.path.join(self.remote_path, 'magvit', 'run_training.py')
        write_script_cmd = f"cd {self.remote_path}/magvit && cat > run_training.py << 'TRAINSCRIPT'\n{training_script}\nTRAINSCRIPT"
        
        # Run training in background with nohup, redirecting output to log file
        # Use nohup to ensure process survives SSH disconnection
        # Redirect both stdout and stderr to log file
        # Create PID file in same directory
        run_cmd = f"cd {self.remote_path}/magvit && nohup python3 run_training.py > {log_file} 2>&1 & echo $! > {pid_file}"
        
        print(f"🚀 Starting training in background (detached session)...")
        print(f"   Log file: {log_file}")
        print(f"   PID file: {pid_file}")
        print(f"   Training will continue even if SSH connection drops")
        
        # Write script first
        write_result = self._run_ssh_command(write_script_cmd, capture_output=True)
        if write_result.returncode != 0:
            print("❌ Failed to write training script!")
            return False
        
        # Run training in background using a more reliable method
        # Create a wrapper script that ensures proper startup and verification
        wrapper_script = f"""#!/bin/bash
cd {self.remote_path}/magvit

# Start training in background
setsid nohup python3 run_training.py > {log_file} 2>&1 < /dev/null &
TRAIN_PID=$!

# Save PID
echo $TRAIN_PID > {pid_file}

# Wait a moment for process to start
sleep 2

# Verify process is actually running
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "STARTED:$TRAIN_PID"
    exit 0
else
    echo "FAILED:Process not running"
    exit 1
fi
"""
        
        # Write wrapper script
        wrapper_path = os.path.join(self.remote_path, 'magvit', 'start_training.sh')
        write_wrapper_cmd = f"cd {self.remote_path}/magvit && cat > start_training.sh << 'WRAPPEREOF'\n{wrapper_script}\nWRAPPEREOF\nchmod +x start_training.sh"
        
        write_wrapper_result = self._run_ssh_command(write_wrapper_cmd, capture_output=True)
        if write_wrapper_result.returncode != 0:
            print("❌ Failed to write wrapper script!")
            return False
        
        # Run wrapper script (this will start training and verify it's running)
        run_cmd = f"cd {self.remote_path}/magvit && bash start_training.sh"
        result = self._run_ssh_command(run_cmd, capture_output=True, timeout=10)
        
        if result.returncode == 0:
            # Parse output to get PID
            stdout_text = result.stdout.decode('utf-8') if isinstance(result.stdout, bytes) else str(result.stdout)
            
            if 'STARTED:' in stdout_text:
                pid = stdout_text.split('STARTED:')[1].strip().split()[0]
                print(f"✅ Training started in background! PID: {pid}")
                print(f"   Process verified as running")
                print(f"   Process will continue even if you disconnect")
                print(f"")
                print(f"📊 Monitoring commands:")
                print(f"   Check status: python3 main_macbook.py --check-training-status")
                print(f"   View logs: python3 main_macbook.py --command 'tail -20 {log_file}'")
                print(f"   Check progress: python3 main_macbook.py --command 'tail -f {log_file}'")
                return True
            else:
                print("❌ Training failed to start!")
                print(f"   Output: {stdout_text}")
                return False
        else:
            print("❌ Failed to start training!")
            stderr_text = result.stderr.decode('utf-8') if (result.stderr and isinstance(result.stderr, bytes)) else str(result.stderr) if result.stderr else "Unknown error"
            print(f"   Error: {stderr_text.strip()}")
            return False
    
    def download_checkpoint_visualizations(self) -> bool:
        """Download checkpoint visualization images (PNG) to MacBook during training"""
        output_dir = os.path.join(self.remote_path, 'magvit/experiments/kinematics_magvit')
        local_path = self.config['project']['local_path']
        local_output_dir = os.path.join(local_path, 'magvit/experiments/kinematics_magvit')
        
        # Create local directory
        os.makedirs(local_output_dir, exist_ok=True)
        
        print(f"📥 Downloading checkpoint visualizations...")
        print(f"   Remote: {output_dir}")
        print(f"   Local: {local_output_dir}")
        
        # Use rsync to download PNG files (images only, not checkpoints)
        rsync_cmd = [
            'rsync',
            '-avz',
            '--progress',
            '--include', '*.png',
            '--include', '*.jpg',
            '--include', '*.jpeg',
            '--exclude', '*',
            '-e', f'ssh -p {self.ssh_port} -o StrictHostKeyChecking=no' + (f' -i {self.ssh_key}' if self.ssh_key else ''),
            f'{self.user}@{self.host}:{output_dir}/',
            f'{local_output_dir}/'
        ]
        
        print(f"   Downloading: {self.host}:{output_dir}/*.png -> {local_output_dir}/")
        
        try:
            result = subprocess.run(rsync_cmd, capture_output=False, timeout=300)
            
            if result.returncode == 0:
                # Count downloaded files
                png_files = list(Path(local_output_dir).glob('*.png'))
                if png_files:
                    print(f"✅ Downloaded {len(png_files)} visualization files")
                    for f in sorted(png_files)[-5:]:  # Show last 5
                        print(f"   - {f.name}")
                else:
                    print(f"⚠️  No PNG files found (training may not have generated visualizations yet)")
                return True
            else:
                print(f"⚠️  Download had issues (files may not exist yet)")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Download timed out!")
            return False
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def check_training_status(self) -> bool:
        """Check status of background training without interfering with it"""
        output_dir = os.path.join(self.remote_path, 'magvit/experiments/kinematics_magvit')
        log_file = os.path.join(output_dir, 'training.log')
        pid_file = os.path.join(output_dir, 'training.pid')
        checkpoint_file = os.path.join(output_dir, 'checkpoint_latest.pth')
        
        print("📊 Checking training status...")
        print("")
        
        # Check if PID file exists
        check_pid_cmd = f"test -f {pid_file} && cat {pid_file} || echo 'NO_PID'"
        pid_result = self._run_ssh_command(check_pid_cmd, capture_output=True)
        pid_text = pid_result.stdout.decode('utf-8') if isinstance(pid_result.stdout, bytes) else str(pid_result.stdout)
        pid = pid_text.strip()
        
        if pid and pid != 'NO_PID':
            # Check if process is running
            ps_cmd = f"ps -p {pid} > /dev/null 2>&1 && echo 'RUNNING' || echo 'NOT_RUNNING'"
            ps_result = self._run_ssh_command(ps_cmd, capture_output=True)
            ps_text = ps_result.stdout.decode('utf-8') if isinstance(ps_result.stdout, bytes) else str(ps_result.stdout)
            is_running = 'RUNNING' in ps_text.strip()
            
            print(f"🔄 Process Status:")
            print(f"   PID: {pid}")
            print(f"   Running: {'✅ Yes' if is_running else '❌ No'}")
        else:
            print(f"🔄 Process Status: ❌ No PID file found")
            is_running = False
        
        print("")
        
        # Check checkpoint
        check_ckpt_cmd = f"test -f {checkpoint_file} && echo 'EXISTS' || echo 'NOT_EXISTS'"
        ckpt_result = self._run_ssh_command(check_ckpt_cmd, capture_output=True)
        ckpt_text = ckpt_result.stdout.decode('utf-8') if isinstance(ckpt_result.stdout, bytes) else str(ckpt_result.stdout)
        has_checkpoint = 'EXISTS' in ckpt_text.strip()
        
        if has_checkpoint:
            # Get checkpoint info (safe - just reading, not interfering)
            # Build command string carefully to avoid variable name conflicts
            ckpt_info_cmd = f"""python3 << 'CMDPYEOF'
import torch
try:
    ckpt = torch.load('{checkpoint_file}', map_location='cpu', weights_only=False)
    epoch_val = ckpt.get('epoch', -1) + 1
    best_loss_val = ckpt.get('best_val_loss', float('inf'))
    print('EPOCH:' + str(epoch_val))
    print('LOSS:' + str(round(best_loss_val, 4)))
except Exception as e:
    print('ERROR:' + str(e))
CMDPYEOF
"""
            ckpt_info_result = self._run_ssh_command(f"cd {output_dir} && {ckpt_info_cmd}", capture_output=True)
            ckpt_info_text = ckpt_info_result.stdout.decode('utf-8') if isinstance(ckpt_info_result.stdout, bytes) else str(ckpt_info_result.stdout)
            
            epoch = None
            loss = None
            for line in ckpt_info_text.strip().split('\n'):
                if line.startswith('EPOCH:'):
                    epoch = line.split('EPOCH:')[1].strip()
                elif line.startswith('LOSS:'):
                    loss = line.split('LOSS:')[1].strip()
            
            print(f"📦 Checkpoint Status:")
            print(f"   Latest checkpoint: ✅ Found")
            if epoch and epoch != 'None':
                try:
                    epoch_num = int(epoch)
                    print(f"   Current epoch: {epoch_num}/400 ({epoch_num/400*100:.1f}%)")
                except (ValueError, TypeError):
                    print(f"   Current epoch: {epoch}")
            if loss and loss != 'None':
                print(f"   Best validation loss: {loss}")
        else:
            print(f"📦 Checkpoint Status: ⏳ No checkpoint yet (training just started)")
        
        print("")
        
        # Check log file
        log_tail_cmd = f"tail -5 {log_file} 2>/dev/null | grep -E 'Epoch|Loss|Val Loss|Summary' | tail -3 || echo 'No recent log entries'"
        log_result = self._run_ssh_command(log_tail_cmd, capture_output=True)
        log_text = log_result.stdout.decode('utf-8') if isinstance(log_result.stdout, bytes) else str(log_result.stdout)
        
        if log_text.strip() and 'No recent log entries' not in log_text:
            print(f"📝 Recent Log Activity:")
            for line in log_text.strip().split('\n')[:3]:
                if line.strip():
                    print(f"   {line.strip()}")
        else:
            print(f"📝 Recent Log Activity: ⏳ No recent activity")
        
        print("")
        
        # Overall status
        # Check if training completed (epoch reached max epochs)
        training_completed = False
        if epoch and epoch != 'None':
            try:
                epoch_num = int(epoch)
                # Training is complete if epoch >= 400 (epochs are 0-indexed, so 400 = completed all 400 epochs)
                # Or if checkpoint shows epoch 399 (which is the 400th epoch, 0-indexed)
                if epoch_num >= 399:  # 0-indexed: epoch 399 = 400th epoch completed
                    training_completed = True
            except (ValueError, TypeError):
                pass
        
        # Check log file for completion message
        completion_check_cmd = f"grep -i 'Training completed\\|Training finished' {log_file} 2>/dev/null | tail -1 || echo ''"
        completion_result = self._run_ssh_command(completion_check_cmd, capture_output=True)
        completion_text = completion_result.stdout.decode('utf-8') if isinstance(completion_result.stdout, bytes) else str(completion_result.stdout)
        if 'completed' in completion_text.lower() or 'finished' in completion_text.lower():
            training_completed = True
        
        if training_completed:
            print("✅ Training completed successfully!")
            print(f"   Final epoch: {epoch if epoch else 'N/A'}/400")
            print(f"   Best validation loss: {loss if loss else 'N/A'}")
            print(f"   Checkpoint saved: {checkpoint_file}")
            print(f"   Ready for inference: python3 main_macbook.py --run-magvit-inference")
        elif is_running:
            print("🔄 Training is running successfully!")
            print(f"   Current epoch: {epoch if epoch else 'In progress...'}/400")
            print(f"   Check again anytime with: python3 main_macbook.py --check-training-status")
        elif has_checkpoint:
            print("⚠️  Training process not running, but checkpoint exists")
            print(f"   Last checkpoint: Epoch {epoch if epoch else 'Unknown'}/400")
            print("   Training may have stopped or encountered an error")
            print("   Check logs: python3 main_macbook.py --command 'tail -50 {log_file}'")
            print("   To resume: python3 main_macbook.py --run-magvit-kinematics")
        else:
            print("❌ Training not running and no checkpoint found")
            print("   Start training: python3 main_macbook.py --run-magvit-kinematics")
        
        return is_running
    
    def run_magvit_inference(self) -> bool:
        """Run MAGVIT inference to demonstrate prediction of missing observations"""
        print("🎬 Running MAGVIT inference example...")
        print("   This will demonstrate predicting missing observations in kinematics videos")
        
        # Find the latest checkpoint
        experiments_dir = os.path.join(self.remote_path, 'magvit/experiments/kinematics_magvit')
        
        # Check for checkpoint - try multiple locations and patterns
        find_checkpoint_cmd = f"find {experiments_dir} -name '*.pth' -o -name '*.pt' 2>/dev/null | head -1"
        checkpoint_result = self._run_ssh_command(find_checkpoint_cmd, capture_output=True)
        stdout_text = checkpoint_result.stdout.decode('utf-8') if isinstance(checkpoint_result.stdout, bytes) else str(checkpoint_result.stdout)
        checkpoint_path = stdout_text.strip()
        
        # If not found, check in nested directories or use kinematics checkpoint
        if not checkpoint_path or checkpoint_path == '':
            # Try kinematics checkpoint first
            find_checkpoint_cmd = f"find {experiments_dir} -name 'checkpoint_best.pth' -o -name 'checkpoint_latest.pth' 2>/dev/null | head -1"
            checkpoint_result = self._run_ssh_command(find_checkpoint_cmd, capture_output=True)
            stdout_text = checkpoint_result.stdout.decode('utf-8') if isinstance(checkpoint_result.stdout, bytes) else str(checkpoint_result.stdout)
            checkpoint_path = stdout_text.strip()
        
        # If still not found, DO NOT fall back to other checkpoints
        # This prevents using wrong checkpoint (e.g., synthetic data checkpoint for kinematics)
        if not checkpoint_path or checkpoint_path == '':
            print("❌ No kinematics training checkpoint found!")
            print(f"   Expected location: {experiments_dir}")
            print(f"   This checkpoint is required for kinematics inference")
            print(f"")
            print("   Please train the model first:")
            print("   python3 main_macbook.py --run-magvit-kinematics")
            print("")
            print("   ⚠️  Do NOT use checkpoints from other training runs (e.g., synthetic data)")
            return False
        
        # Make path relative or absolute as needed
        if checkpoint_path.startswith('/'):
            # Already absolute
            full_checkpoint_path = checkpoint_path
        else:
            # Relative path
            full_checkpoint_path = os.path.join(experiments_dir, checkpoint_path)
        
        # Use a sample kinematics CSV for demonstration
        sample_csv = os.path.join(self.remote_path, 'kinematics_examples/example_01_constant_velocity.csv')
        
        # Run inference
        output_dir = os.path.join(self.remote_path, 'magvit/inference_results')
        command = f"cd {self.remote_path}/magvit && python3 inference_example.py "
        command += f"--checkpoint {full_checkpoint_path} "
        command += f"--input {sample_csv} "
        command += f"--input_type csv "
        command += f"--mask_ratio 0.5 "
        command += f"--output_dir {output_dir} "
        command += "--device cuda"
        
        result = self._run_ssh_command(command, capture_output=False)
        
        if result.returncode == 0:
            print("✅ MAGVIT inference completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
            
            # List output files
            list_cmd = f"cd {self.remote_path}/magvit && ls -la {output_dir}/"
            self._run_ssh_command(list_cmd)
            return True
        else:
            print("❌ Failed to run MAGVIT inference!")
            return False
    
    def install_magvit_dependencies(self) -> bool:
        """Install MAGVIT dependencies (PyTorch-based) on vast.ai instance"""
        print("📦 Installing MAGVIT dependencies (PyTorch-based)...")
        print("   This may take several minutes...")
        
        # Install PyTorch-based dependencies (much simpler than JAX/TensorFlow!)
        commands = [
            f"cd {self.remote_path}/magvit",
            # Install PyTorch with CUDA support
            "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            # Install other dependencies
            "pip3 install opencv-python numpy matplotlib tqdm",
            # Optional: tensorboard for logging
            "pip3 install tensorboard"
        ]
        
        # Install dependencies
        for i, cmd in enumerate(commands, 1):
            print(f"   Step {i}/{len(commands)}: {' '.join(cmd.split()[:3])}...")
            result = self._run_ssh_command(cmd, capture_output=False)
            
            if result.returncode != 0:
                print(f"   ⚠️  Step {i} had non-zero exit code (may be OK if packages already installed)")
        
        # Verify installation
        print("\n🔍 Verifying installation...")
        verify_cmd = f"cd {self.remote_path}/magvit && python3 -c \"import torch; import cv2; import numpy; import matplotlib; from simple_magvit_model import SimpleMagvitModel, ModelConfig; print('✅ All dependencies: OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())\""
        verify_result = self._run_ssh_command(verify_cmd, capture_output=True)
        
        if verify_result.returncode == 0:
            print("✅ MAGVIT dependencies installed successfully!")
            stdout_text = verify_result.stdout.decode('utf-8') if isinstance(verify_result.stdout, bytes) else str(verify_result.stdout)
            print(stdout_text.strip())
            return True
        else:
            print("⚠️  Some dependencies may not be installed correctly")
            stderr_text = verify_result.stderr.decode('utf-8') if (verify_result.stderr and isinstance(verify_result.stderr, bytes)) else str(verify_result.stderr) if verify_result.stderr else "Unknown error"
            print("   Error:", stderr_text.strip())
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Cyber Evan Project - Vast.ai Connector')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    parser.add_argument('--setup', action='store_true', help='Setup environment')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--sync', action='store_true', help='Sync project files to vast.ai')
    parser.add_argument('--download-output', action='store_true', help='Download output files from vast.ai')
    parser.add_argument('--shell', action='store_true', help='Open remote shell')
    parser.add_argument('--command', help='Execute remote command')
    parser.add_argument('--run-kinematics', action='store_true', help='Run kinematics examples generation')
    parser.add_argument('--run-magvit', action='store_true', help='Run MAGVIT simple example')
    parser.add_argument('--run-magvit-training', action='store_true', help='Run MAGVIT training and inference example')
    parser.add_argument('--convert-kinematics', action='store_true', help='Convert kinematics CSV examples to video dataset')
    parser.add_argument('--run-magvit-kinematics', action='store_true', help='Run MAGVIT training on kinematics dataset (runs in background)')
    parser.add_argument('--check-training-status', action='store_true', help='Check status of background training (safe, non-interfering)')
    parser.add_argument('--download-checkpoint-images', action='store_true', help='Download checkpoint visualization images to MacBook')
    parser.add_argument('--run-magvit-inference', action='store_true', help='Run MAGVIT inference to predict missing observations')
    parser.add_argument('--install-magvit-deps', action='store_true', help='Install MAGVIT dependencies on vast.ai')
    
    args = parser.parse_args()
    
    # Initialize connector
    connector = VastAIConnector(args.config)
    
    print("🚀 Cyber Evan Project - Vast.ai Connector")
    print("=" * 50)
    
    # Test connection first
    if not connector.test_connection():
        print("❌ Cannot proceed without connection. Please check your configuration.")
        sys.exit(1)
    
    # Execute requested operations
    if args.test:
        print("✅ Connection test completed!")
        return
    
    if args.setup:
        if not connector.setup_environment():
            sys.exit(1)
    
    if args.install_deps:
        if not connector.install_system_dependencies():
            sys.exit(1)
        if not connector.install_python_dependencies():
            sys.exit(1)
    
    if args.sync:
        if not connector.sync_project_files():
            sys.exit(1)
    
    if args.download_output:
        if not connector.download_output_files():
            sys.exit(1)
    
    if args.command:
        connector.execute_remote_command(args.command)
    
    if args.shell:
        connector.get_remote_shell()
    
    if args.run_kinematics:
        if not connector.run_kinematics_examples():
            sys.exit(1)
    
    if args.run_magvit:
        if not connector.run_magvit_example():
            sys.exit(1)
    
    if args.run_magvit_training:
        if not connector.run_magvit_training():
            sys.exit(1)
    
    if args.convert_kinematics:
        if not connector.convert_kinematics_to_video():
            sys.exit(1)
    
    if args.run_magvit_kinematics:
        if not connector.run_magvit_kinematics_training():
            sys.exit(1)
    
    if args.check_training_status:
        connector.check_training_status()
        sys.exit(0)
    
    if args.download_checkpoint_images:
        if not connector.download_checkpoint_visualizations():
            sys.exit(1)
        sys.exit(0)
    
    if args.run_magvit_inference:
        if not connector.run_magvit_inference():
            sys.exit(1)
    
    if args.install_magvit_deps:
        if not connector.install_magvit_dependencies():
            sys.exit(1)
    
    # If no specific action requested, show help
    if not any([args.setup, args.install_deps, args.sync, args.download_output, args.command, args.shell, args.run_kinematics, args.run_magvit, args.run_magvit_training, args.install_magvit_deps]):
        print("\n📋 Available operations:")
        print("   --test          Test connection to vast.ai")
        print("   --setup         Setup basic environment")
        print("   --install-deps  Install all dependencies")
        print("   --sync          Sync project files to vast.ai")
        print("   --download-output Download output files from vast.ai")
        print("   --shell         Open remote shell")
        print("   --command CMD   Execute remote command")
        print("   --run-kinematics Run kinematics examples generation")
        print("   --run-magvit     Run MAGVIT simple example")
        print("   --run-magvit-training Run MAGVIT training and inference example")
        print("   --install-magvit-deps Install MAGVIT dependencies on vast.ai")
        print("\n💡 Example: python3 main_macbook.py --install-deps --sync")
        print("💡 Example: python3 main_macbook.py --run-kinematics")
        print("💡 Example: python3 main_macbook.py --install-magvit-deps")
        print("💡 Example: python3 main_macbook.py --run-magvit-training")
        print("💡 Example: python3 main_macbook.py --run-magvit")
        print("💡 Example: python3 main_macbook.py --download-output")


if __name__ == "__main__":
    main()
