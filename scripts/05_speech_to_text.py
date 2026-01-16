#!/usr/bin/env python3
"""
Speech-to-Text with Whisper Large-v3
Transcribe audio files to text
"""

import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path


def load_whisper(model_size="large-v3"):
    """
    Load Whisper model

    Args:
        model_size: "tiny", "base", "small", "medium", "large-v3"
    """
    model_id = f"openai/whisper-{model_size}"
    print(f"Loading Whisper {model_size}...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, processor


def transcribe(
    model,
    processor,
    audio_path,
    language="en",
    task="transcribe"
):
    """
    Transcribe audio file to text

    Args:
        model: Whisper model
        processor: Whisper processor
        audio_path: Path to audio file
        language: Language code (e.g., "en", "es", "fr")
        task: "transcribe" or "translate" (to English)

    Returns:
        Transcribed text
    """
    print(f"Transcribing: {audio_path}")

    # Load audio at 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s")

    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to("cuda", dtype=torch.float16)

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            max_new_tokens=448,
            language=language,
            task=task
        )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


if __name__ == "__main__":
    model, processor = load_whisper("large-v3")

    # Example: Transcribe an audio file
    audio_path = "/home/satish/test_speech.wav"

    if Path(audio_path).exists():
        text = transcribe(model, processor, audio_path)
        print("\nTranscription:")
        print("-" * 50)
        print(text)
    else:
        print(f"Audio file not found: {audio_path}")
        print("Provide an audio file to transcribe")
