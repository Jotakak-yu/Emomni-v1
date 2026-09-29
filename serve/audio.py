"""Inference audio loading without training-data dependencies."""
import subprocess

import numpy as np


def load_audio(path, sample_rate=16000, normalize=True):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", "30.001",
         "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
        capture_output=True, check=True, timeout=30,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("invalid_audio")
    if samples.size > sample_rate * 30:
        raise ValueError("audio_too_long")
    peak = np.abs(samples).max()
    if normalize and peak:
        samples /= peak
    return samples
