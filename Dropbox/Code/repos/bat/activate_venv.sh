#!/bin/bash
# Cyber Evan Virtual Environment Activation Script
# This script activates the Python virtual environment for the Cyber Evan project

echo "🚀 Activating Cyber Evan Virtual Environment..."
echo "=================================================="

# Navigate to project directory
cd /Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/cyber_evan

# Activate virtual environment
source ~/cyber_evan_venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📍 Python location: $(which python)"
echo "📍 Python version: $(python --version)"
echo "📍 Working directory: $(pwd)"
echo ""
echo "🔧 Available commands:"
echo "   python main_macbook.py --test          # Test connection to vast.ai"
echo "   python main_macbook.py --sync          # Sync files to vast.ai"
echo "   python main_macbook.py --shell         # Open remote shell"
echo "   python main_macbook.py --command 'cmd' # Execute remote command"
echo ""
echo "💡 To deactivate: type 'deactivate'"
echo "=================================================="

