#!/bin/bash
# RunPod Setup Script for LLaDA-1.5
# Run this script after connecting to your RunPod instance

set -e

echo "=========================================="
echo "Setting up LLaDA-1.5 on RunPod"
echo "=========================================="

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install torch transformers accelerate huggingface_hub

# Clone the repo if not already present
if [ ! -d "LLaDA-1.5" ]; then
    echo ""
    echo "Cloning LLaDA-1.5 repository..."
    git clone https://github.com/junos-ai-org/LLaDA-1.5.git
    cd LLaDA-1.5
else
    echo "LLaDA-1.5 directory already exists"
    cd LLaDA-1.5 2>/dev/null || true
fi

# Pre-download the model (optional but recommended)
echo ""
echo "Do you want to pre-download the model? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Pre-downloading LLaDA-1.5 model (~16GB)..."
    python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True)
print('Downloading model weights...')
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True)
print('Model downloaded successfully!')
"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To run inference:"
echo "  python inference.py --interactive"
echo "  python inference.py --prompt 'Your question here'"
echo "=========================================="
