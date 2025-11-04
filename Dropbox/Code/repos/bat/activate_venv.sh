#!/bin/bash
# Bat Project Virtual Environment Activation Script
# This script activates the Python virtual environment for the bat project
# Prevents accidental package installation on MacBook

echo "🚀 Activating Bat Project Virtual Environment..."
echo "=================================================="

# Navigate to project directory
cd /Users/mike/Dropbox/Code/repos/bat

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment (first time setup)..."
    python3 -m venv venv
    echo "✅ Virtual environment created!"
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📍 Python location: $(which python3)"
echo "📍 Python version: $(python3 --version)"
echo "📍 Working directory: $(pwd)"
echo ""
echo "🔧 Available commands:"
echo "   python3 main_macbook.py --test          # Test connection to vast.ai"
echo "   python3 main_macbook.py --sync          # Sync files to vast.ai"
echo "   python3 main_macbook.py --shell         # Open remote shell"
echo "   python3 main_macbook.py --command 'cmd' # Execute remote command"
echo ""
echo "💡 To deactivate: type 'deactivate'"
echo "⚠️  Remember: All computation packages should be installed on vast.ai, not MacBook"
echo "=================================================="
