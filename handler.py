"""
RunPod Serverless Handler for LLaDA-1.5

This handler allows you to deploy LLaDA-1.5 as a serverless endpoint on RunPod.

Usage:
1. Build Docker image with this handler
2. Push to Docker Hub
3. Create Serverless Endpoint on RunPod using your image
"""

import os
import runpod
import torch
from transformers import AutoModel, AutoTokenizer

# Configure cache directory for network volume (if available)
CACHE_DIR = os.environ.get('HF_HOME', '/workspace/huggingface')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'hub')

# Global model and tokenizer (loaded once at cold start)
model = None
tokenizer = None


def load_model():
    """Load model on cold start."""
    global model, tokenizer

    if model is None:
        print(f"Loading LLaDA-1.5 model (cache: {CACHE_DIR})...")
        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-1.5',
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            'GSAI-ML/LLaDA-1.5',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()
        print("Model loaded successfully!")

    return model, tokenizer


def handler(job):
    """
    RunPod serverless handler function.

    Expected input format:
    {
        "input": {
            "prompt": "Your question here",
            "max_tokens": 256  # optional
        }
    }
    """
    job_input = job.get("input", {})

    # Get parameters
    prompt = job_input.get("prompt")
    max_tokens = job_input.get("max_tokens", 256)

    if not prompt:
        return {"error": "No prompt provided"}

    # Load model (cached after first call)
    model, tokenizer = load_model()

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").cuda()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "response": response,
        "prompt": prompt,
        "tokens_generated": len(outputs[0])
    }


# Start the serverless worker
runpod.serverless.start({"handler": handler})
