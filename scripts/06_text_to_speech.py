#!/usr/bin/env python3
"""
Text-to-Speech with Bark and Parler-TTS
Generate natural speech from text
"""

import torch
import numpy as np
from transformers import BarkModel, AutoProcessor
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import scipy.io.wavfile as wavfile
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_bark():
    """Load Bark TTS model (2.1GB VRAM)"""
    print("Loading Bark TTS...")
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16)
    model.to("cuda")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, processor


def load_parler():
    """Load Parler-TTS model (3.3GB VRAM)"""
    print("Loading Parler-TTS...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1"
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, tokenizer


def generate_speech_bark(
    model,
    processor,
    text,
    voice_preset="v2/en_speaker_6",
    output_name="bark_output.wav"
):
    """
    Generate speech with Bark

    Args:
        model: Bark model
        processor: Bark processor
        text: Text to speak (can include [laughs], [sighs], etc.)
        voice_preset: Voice preset (e.g., "v2/en_speaker_6", "v2/en_speaker_9")
        output_name: Output filename
    """
    print(f"Generating: \"{text[:50]}...\"")

    inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        audio_array = model.generate(**inputs)

    audio_array = audio_array.cpu().numpy().squeeze().astype(np.float32)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    sample_rate = model.generation_config.sample_rate
    output_path = OUTPUT_DIR / output_name
    wavfile.write(str(output_path), rate=sample_rate, data=audio_int16)

    duration = len(audio_array) / sample_rate
    print(f"Saved: {output_path} ({duration:.1f}s)")
    return output_path


def generate_speech_parler(
    model,
    tokenizer,
    text,
    description="A female speaker delivers a clear and expressive speech with moderate speed.",
    output_name="parler_output.wav"
):
    """
    Generate speech with Parler-TTS

    Args:
        model: Parler-TTS model
        tokenizer: Parler-TTS tokenizer
        text: Text to speak
        description: Voice/style description in natural language
        output_name: Output filename
    """
    print(f"Generating: \"{text[:50]}...\"")
    print(f"Style: {description[:50]}...")

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to("cuda")
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

    audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
    audio_int16 = (audio_arr * 32767).astype(np.int16)

    output_path = OUTPUT_DIR / output_name
    wavfile.write(str(output_path), rate=model.config.sampling_rate, data=audio_int16)

    duration = len(audio_arr) / model.config.sampling_rate
    print(f"Saved: {output_path} ({duration:.1f}s)")
    return output_path


if __name__ == "__main__":
    # Example with Bark
    print("=== Bark TTS ===")
    bark_model, bark_processor = load_bark()
    generate_speech_bark(
        bark_model, bark_processor,
        "Hello! Welcome to the AI toolkit. [laughs] This is amazing!",
        voice_preset="v2/en_speaker_6",
        output_name="bark_welcome.wav"
    )

    # Clear VRAM
    del bark_model, bark_processor
    torch.cuda.empty_cache()

    # Example with Parler-TTS
    print("\n=== Parler-TTS ===")
    parler_model, parler_tokenizer = load_parler()
    generate_speech_parler(
        parler_model, parler_tokenizer,
        "Hello! Welcome to the AI toolkit. This system can generate natural speech from text.",
        description="A male speaker with a deep voice speaks slowly and clearly.",
        output_name="parler_welcome.wav"
    )
