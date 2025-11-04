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
    
    def _run_ssh_command(self, command: str, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Execute SSH command on vast.ai instance"""
        ssh_cmd = ['ssh', '-p', str(self.ssh_port), '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=30']
        
        # Add SSH key if provided
        if self.ssh_key:
            ssh_cmd.extend(['-i', self.ssh_key])
        
        ssh_cmd.extend([f'{self.user}@{self.host}', command])
        
        print(f"🔗 Executing: {' '.join(ssh_cmd[:3])} ... {command}")
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=capture_output,
                text=True,
                timeout=60
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
