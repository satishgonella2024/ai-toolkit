#!/usr/bin/env python3
"""
Image Captioning and Visual Q&A with BLIP and LLaVA
"""

import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)
from PIL import Image
from pathlib import Path


def load_blip():
    """Load BLIP for fast image captioning (0.9GB VRAM)"""
    print("Loading BLIP Large...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16
    ).to("cuda")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, processor


def load_llava():
    """Load LLaVA for detailed image understanding (4.4GB VRAM with 4-bit)"""
    print("Loading LLaVA-v1.6...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, processor


def caption_blip(model, processor, image_path, prompt=None):
    """
    Generate caption with BLIP

    Args:
        model: BLIP model
        processor: BLIP processor
        image_path: Path to image
        prompt: Optional prompt (e.g., "a photograph of")
    """
    image = Image.open(image_path).convert("RGB")

    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to("cuda", torch.float16)
    else:
        inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)

    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def ask_llava(model, processor, image_path, question):
    """
    Ask a question about an image with LLaVA

    Args:
        model: LLaVA model
        processor: LLaVA processor
        image_path: Path to image
        question: Question about the image
    """
    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda")

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=True)

    # Extract assistant's response
    answer = response.split("[/INST]")[-1].strip()
    return answer


if __name__ == "__main__":
    # Test image
    image_path = "/home/satish/sdxl_full_output.png"

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Run 02_image_generation.py first")
        exit(1)

    # BLIP captioning
    print("=== BLIP Captioning ===")
    blip_model, blip_processor = load_blip()
    caption = caption_blip(blip_model, blip_processor, image_path)
    print(f"Caption: {caption}\n")

    # Clear VRAM
    del blip_model, blip_processor
    torch.cuda.empty_cache()

    # LLaVA Q&A
    print("=== LLaVA Visual Q&A ===")
    llava_model, llava_processor = load_llava()

    questions = [
        "Describe this image in detail.",
        "What is the mood or atmosphere of this scene?",
        "What time of day does this appear to be?",
    ]

    for question in questions:
        print(f"Q: {question}")
        answer = ask_llava(llava_model, llava_processor, image_path, question)
        print(f"A: {answer}\n")
