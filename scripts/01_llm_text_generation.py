#!/usr/bin/env python3
"""
LLM Text Generation with Mistral-7B, Llama-3.1-8B, and more
Uses 4-bit quantization to fit in 16GB VRAM
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name="mistral"):
    """Load a quantized LLM model"""

    models = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama": "meta-llama/Llama-3.1-8B-Instruct",  # Official Meta (requires HF approval)
        "llama-nous": "NousResearch/Meta-Llama-3-8B-Instruct",  # Community version (no approval needed)
        "gpt2": "gpt2",
    }

    model_id = models.get(model_name, models["mistral"])
    print(f"Loading {model_id}...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens=200):
    """Generate text from a prompt"""

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Example usage
    model, tokenizer = load_model("mistral")

    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        response = generate_text(model, tokenizer, prompt)
        print(response)
