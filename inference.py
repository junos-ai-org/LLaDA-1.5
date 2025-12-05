#!/usr/bin/env python3
"""
LLaDA-1.5 Inference Script for RunPod

This script provides a simple interface to run inference with LLaDA-1.5,
a large diffusion language model trained with variance-reduced preference optimization.

Usage:
    python inference.py --prompt "Your question here"
    python inference.py --interactive  # For interactive chat mode
    python inference.py --show-unmasking  # Show the diffusion unmasking process

Environment Variables:
    HF_HOME: Set to your network volume path (e.g., /runpod-volume/huggingface)
"""

import argparse
import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer


def setup_cache_dir(cache_dir=None):
    """Configure HuggingFace cache directory for network volume storage."""
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'hub')
        print(f"Using cache directory: {cache_dir}")
        # Create directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    elif os.environ.get('HF_HOME'):
        print(f"Using cache directory from HF_HOME: {os.environ['HF_HOME']}")


def load_model(device="cuda", cache_dir=None):
    """Load the LLaDA-1.5 model and tokenizer."""
    setup_cache_dir(cache_dir)

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


def generate_response(model, tokenizer, prompt, max_new_tokens=256, show_unmasking=False, steps=64):
    """
    Generate a response for the given prompt.

    Args:
        model: The LLaDA model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        show_unmasking: If True, display the iterative unmasking process
        steps: Number of diffusion steps (more steps = higher quality but slower)
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        if show_unmasking:
            # Use custom generation to show unmasking process
            response = generate_with_visualization(
                model, tokenizer, inputs,
                max_new_tokens=max_new_tokens,
                steps=steps
            )
        else:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def generate_with_visualization(model, tokenizer, inputs, max_new_tokens=256, steps=64):
    """
    Generate text while visualizing the diffusion unmasking process.

    LLaDA uses masked diffusion - tokens start masked and are progressively revealed.
    This function shows the intermediate states during generation.
    """
    print("\n" + "="*60)
    print("Diffusion Unmasking Process")
    print("="*60)

    # Get the mask token
    mask_token = tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else '[MASK]'
    mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.convert_tokens_to_ids(mask_token)

    # Check if model has a method to expose intermediate states
    if hasattr(model, 'generate_with_callback'):
        # If model supports callbacks, use them
        intermediate_states = []

        def callback(step, tokens):
            intermediate_states.append(tokens.clone())
            text = tokenizer.decode(tokens[0], skip_special_tokens=False)
            # Replace mask tokens with underscores for visibility
            text = text.replace(mask_token, '____')
            print(f"Step {step:3d}: {text[:100]}...")

        outputs = model.generate_with_callback(
            **inputs,
            max_new_tokens=max_new_tokens,
            callback=callback
        )
    elif hasattr(model, 'diffusion_generate'):
        # Alternative method name
        outputs = model.diffusion_generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            steps=steps,
            verbose=True
        )
    else:
        # Fallback: standard generation with step-by-step output attempt
        print(f"Note: Using standard generation. The model may not expose")
        print(f"intermediate diffusion states through the generate() API.")
        print(f"Check the model's documentation for diffusion-specific methods.")
        print("-"*60)

        # Try to use any available diffusion parameters
        gen_kwargs = {
            **inputs,
            'max_new_tokens': max_new_tokens,
        }

        # Check for diffusion-specific parameters the model might accept
        if hasattr(model.config, 'diffusion_steps'):
            gen_kwargs['steps'] = steps

        outputs = model.generate(**gen_kwargs)

    print("="*60)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def interactive_mode(model, tokenizer, show_unmasking=False, steps=64):
    """Run in interactive chat mode."""
    print("\n" + "="*50)
    print("LLaDA-1.5 Interactive Mode")
    print("Type 'quit' or 'exit' to end the session")
    if show_unmasking:
        print("Unmasking visualization: ENABLED")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            if not show_unmasking:
                print("LLaDA-1.5: ", end="", flush=True)

            response = generate_response(
                model, tokenizer, user_input,
                show_unmasking=show_unmasking,
                steps=steps
            )

            if show_unmasking:
                print(f"\nFinal response: {response}")
            else:
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
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="HuggingFace cache directory (e.g., /runpod-volume/huggingface)"
    )
    parser.add_argument(
        "--show-unmasking",
        action="store_true",
        help="Show the diffusion unmasking process during generation"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of diffusion steps (default: 64, more = better quality but slower)"
    )

    args = parser.parse_args()

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Inference will be very slow.")

    device = "cpu" if args.cpu else "cuda"
    model, tokenizer = load_model(device, cache_dir=args.cache_dir)

    if args.interactive:
        interactive_mode(model, tokenizer, show_unmasking=args.show_unmasking, steps=args.steps)
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}\n")
        response = generate_response(
            model, tokenizer, args.prompt, args.max_tokens,
            show_unmasking=args.show_unmasking, steps=args.steps
        )
        print(f"Response: {response}")
    else:
        # Default demo
        print("\nRunning demo inference...")
        demo_prompt = "Explain what a diffusion language model is in simple terms."
        print(f"Prompt: {demo_prompt}\n")
        response = generate_response(
            model, tokenizer, demo_prompt, args.max_tokens,
            show_unmasking=args.show_unmasking, steps=args.steps
        )
        print(f"Response: {response}")
        print("\nTip: Use --interactive for chat mode or --prompt for single queries")
        print("      Use --show-unmasking to see the diffusion process")


if __name__ == "__main__":
    main()
