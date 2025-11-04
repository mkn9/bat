#!/usr/bin/env python3
"""
Cyber Evan Project - Main MacBook Interface
Connects to vast.ai instance and manages project execution

Usage:
    python main_macbook.py [--config config.yaml] [--setup] [--test] [--install-deps]
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Cyber Evan Project - Vast.ai Connector')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    parser.add_argument('--setup', action='store_true', help='Setup environment')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--sync', action='store_true', help='Sync project files')
    parser.add_argument('--shell', action='store_true', help='Open remote shell')
    parser.add_argument('--command', help='Execute remote command')
    
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
    
    if args.command:
        connector.execute_remote_command(args.command)
    
    if args.shell:
        connector.get_remote_shell()
    
    # If no specific action requested, show help
    if not any([args.setup, args.install_deps, args.sync, args.command, args.shell]):
        print("\n📋 Available operations:")
        print("   --test          Test connection to vast.ai")
        print("   --setup         Setup basic environment")
        print("   --install-deps  Install all dependencies")
        print("   --sync          Sync project files")
        print("   --shell         Open remote shell")
        print("   --command CMD   Execute remote command")
        print("\n💡 Example: python main_macbook.py --install-deps --sync")


if __name__ == "__main__":
    main()
