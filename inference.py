#!/usr/bin/env python3
"""
LLaDA-1.5 Inference Script for RunPod

This script provides a simple interface to run inference with LLaDA-1.5,
a large diffusion language model trained with variance-reduced preference optimization.

Usage:
    python inference.py --prompt "Your question here"
    python inference.py --interactive  # For interactive chat mode
"""

import argparse
import torch
from transformers import AutoModel, AutoTokenizer


def load_model(device="cuda"):
    """Load the LLaDA-1.5 model and tokenizer."""
    print("Loading LLaDA-1.5 model from HuggingFace...")
    print("This may take a few minutes on first run (downloading ~16GB)...")

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-1.5',
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-1.5',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Model loaded on CPU (this will be slow)")

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response for the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")

    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def interactive_mode(model, tokenizer):
    """Run in interactive chat mode."""
    print("\n" + "="*50)
    print("LLaDA-1.5 Interactive Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("LLaDA-1.5: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="LLaDA-1.5 Inference Script")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate response for"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (not recommended, very slow)"
    )

    args = parser.parse_args()

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Inference will be very slow.")

    device = "cpu" if args.cpu else "cuda"
    model, tokenizer = load_model(device)

    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}\n")
        response = generate_response(model, tokenizer, args.prompt, args.max_tokens)
        print(f"Response: {response}")
    else:
        # Default demo
        print("\nRunning demo inference...")
        demo_prompt = "Explain what a diffusion language model is in simple terms."
        print(f"Prompt: {demo_prompt}\n")
        response = generate_response(model, tokenizer, demo_prompt, args.max_tokens)
        print(f"Response: {response}")
        print("\nTip: Use --interactive for chat mode or --prompt for single queries")


if __name__ == "__main__":
    main()
