#!/usr/bin/env python3
"""
Music Generation with MusicGen
Generate music from text descriptions
"""

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_musicgen(size="medium"):
    """
    Load MusicGen model

    Args:
        size: "small" (2.2GB VRAM), "medium" (7.5GB VRAM), or "large"
    """
    model_id = f"facebook/musicgen-{size}"
    print(f"Loading MusicGen {size}...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model.to("cuda")

    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, processor


def generate_music(
    model,
    processor,
    prompt,
    duration_seconds=10,
    output_name="music_output.wav"
):
    """
    Generate music from a text prompt

    Args:
        model: MusicGen model
        processor: MusicGen processor
        prompt: Text description of the music
        duration_seconds: Length of audio (approximate)
        output_name: Output filename
    """
    print(f"Generating: \"{prompt}\"")

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # ~512 tokens ≈ 10 seconds
    max_tokens = int(duration_seconds * 51.2)

    audio_values = model.generate(**inputs, max_new_tokens=max_tokens)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].cpu().numpy()

    output_path = OUTPUT_DIR / output_name
    wavfile.write(str(output_path), rate=sampling_rate, data=audio_data)

    actual_duration = len(audio_data) / sampling_rate
    print(f"Saved: {output_path} ({actual_duration:.1f}s)")
    return output_path


if __name__ == "__main__":
    model, processor = load_musicgen("medium")

    prompts = [
        ("epic cinematic orchestral music with drums and strings", "epic_orchestral.wav"),
        ("lo-fi hip hop beats, chill relaxing study music", "lofi_beats.wav"),
        ("upbeat electronic dance music with heavy bass", "edm_track.wav"),
        ("acoustic guitar folk song, warm and melodic", "folk_acoustic.wav"),
    ]

    for prompt, filename in prompts:
        generate_music(model, processor, prompt, duration_seconds=10, output_name=filename)
        print()
