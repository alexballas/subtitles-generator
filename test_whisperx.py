#!/usr/bin/env python3

import argparse
import os
import torch
import typing
from omegaconf import ListConfig, DictConfig
from omegaconf.base import ContainerMetadata

# Add safe globals for PyTorch 2.6+ weights_only loading
torch.serialization.add_safe_globals([
    ListConfig,
    DictConfig,
    ContainerMetadata,
    typing.Any,
    typing.Union,
    typing.Optional,
    typing.Dict,
    typing.List,
])

# Patch torch.load to disable weights_only
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

import whisperx

def run_smoke_test(video_path: str | None) -> int:
    print("Testing whisperx basic functionality...")

    try:
        print("Loading whisperx model 'base' on cpu...")
        model = whisperx.load_model("base", "cpu", compute_type="int8")
        print("✓ Model loaded successfully!")

        if not video_path:
            print("No test video provided. Model load check passed.")
            return 0

        if not os.path.exists(video_path):
            print(f"Test video file not found: {video_path}")
            return 1

        print(f"Testing audio loading from: {video_path}")
        audio = whisperx.load_audio(video_path)
        print("✓ Audio loaded successfully!")

        print("Testing transcription...")
        result = model.transcribe(audio, batch_size=1)
        print("✓ Transcription successful!")
        print(f"Detected language: {result['language']}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local whisperx smoke test. Optionally pass a video path."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=os.getenv("WHISPERX_TEST_VIDEO"),
        help="Optional video path. Defaults to WHISPERX_TEST_VIDEO when set.",
    )
    raise SystemExit(run_smoke_test(parser.parse_args().video_path))
