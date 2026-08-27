"""
audio_analysis.py
Shared audio-loading and analysis helpers for both visualizers.
Uses only numpy/scipy + ffmpeg (as a subprocess) -- no librosa dependency,
so it stays lightweight and installs cleanly on a plain Windows Python setup.
"""
import subprocess
import tempfile
import os
import numpy as np
from scipy.io import wavfile


def load_audio(path, target_sr=44100):
    """
    Decode ANY input audio/video file to mono float32 PCM at target_sr using ffmpeg.
    Returns (y, sr) where y is a 1D float32 array in [-1, 1].
    """
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "decoded.wav")
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ac", "1",              # mono
            "-ar", str(target_sr),   # resample
            "-vn",
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode '{path}':\n{result.stderr[-2000:]}")
        sr, y = wavfile.read(wav_path)

    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
    else:
        y = y.astype(np.float32)

    return y, sr


def _hz_to_band_edges(n_bands, fmin, fmax):
    """Log-spaced band edges between fmin and fmax (typical spectrum-analyzer banding)."""
    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    return edges


def compute_spectrum_frames(y, sr, fps, n_bands=48, fmin=30, fmax=16000,
                             window_seconds=1.0 / 30.0, attack=0.6, release=0.15,
                             gamma=0.6):
    """
    Produce a per-video-frame band spectrum, smoothed for pleasant bar motion.

    Returns: array of shape (num_frames, n_bands), values normalized to ~[0, 1].

    attack/release: 0..1 smoothing coefficients (higher = snappier).
        attack controls how fast a band can rise, release how fast it falls
        (classic peak-meter ballistics, like the AudioSpectrumVisualizer bars).
    gamma: <1 boosts quiet detail, >1 emphasizes only loud peaks.
    """
    duration = len(y) / sr
    num_frames = int(np.ceil(duration * fps))
    win_len = max(512, int(sr * window_seconds))
    win_len = 1 << (win_len - 1).bit_length()  # round up to power of 2 for fast FFT
    window = np.hanning(win_len)

    edges = _hz_to_band_edges(n_bands, fmin, fmax)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / sr)
    bin_idx = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0] for i in range(n_bands)]

    raw = np.zeros((num_frames, n_bands), dtype=np.float32)
    half = win_len // 2
    for f in range(num_frames):
        center = int((f / fps) * sr)
        start = center - half
        end = start + win_len
        chunk = np.zeros(win_len, dtype=np.float32)
        src_start = max(0, start)
        src_end = min(len(y), end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        if src_end > src_start:
            chunk[dst_start:dst_end] = y[src_start:src_end]
        spec = np.abs(np.fft.rfft(chunk * window))
        for b in range(n_bands):
            idx = bin_idx[b]
            raw[f, b] = spec[idx].mean() if len(idx) else 0.0

    # normalize per-band-set globally, then apply perceptual gamma curve
    raw = raw / (raw.max() + 1e-9)
    raw = np.power(raw, gamma)

    # attack/release smoothing across time for natural bar movement
    smoothed = np.zeros_like(raw)
    prev = np.zeros(n_bands, dtype=np.float32)
    for f in range(num_frames):
        target = raw[f]
        coef = np.where(target > prev, attack, release)
        prev = prev + coef * (target - prev)
        smoothed[f] = prev

    return smoothed


def compute_waveform_frames(y, sr, fps, window_seconds=0.05):
    """
    Produce a per-frame short window of raw samples for an oscilloscope-style
    waveform visualizer. Returns list of 1D arrays (length = window_seconds*sr).
    """
    duration = len(y) / sr
    num_frames = int(np.ceil(duration * fps))
    win_len = max(64, int(sr * window_seconds))
    frames = []
    half = win_len // 2
    for f in range(num_frames):
        center = int((f / fps) * sr)
        start = center - half
        end = start + win_len
        chunk = np.zeros(win_len, dtype=np.float32)
        src_start = max(0, start)
        src_end = min(len(y), end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        if src_end > src_start:
            chunk[dst_start:dst_end] = y[src_start:src_end]
        frames.append(chunk)
    return frames


def compute_onset_envelope(y, sr, fps):
    """
    Lightweight spectral-flux onset/beat-strength envelope, one value per video frame,
    normalized to [0, 1]. Good enough to drive "pops" in the classic visualizer without
    needing librosa's beat tracker.
    """
    hop = max(1, int(sr / fps))
    win_len = 2048
    window = np.hanning(win_len)
    num_frames = int(np.ceil(len(y) / hop))

    prev_mag = None
    flux = np.zeros(num_frames, dtype=np.float32)
    for f in range(num_frames):
        start = f * hop - win_len // 2
        end = start + win_len
        chunk = np.zeros(win_len, dtype=np.float32)
        src_start = max(0, start)
        src_end = min(len(y), end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        if src_end > src_start:
            chunk[dst_start:dst_end] = y[src_start:src_end]
        mag = np.abs(np.fft.rfft(chunk * window))
        if prev_mag is not None:
            diff = mag - prev_mag
            flux[f] = np.sum(np.maximum(diff, 0))
        prev_mag = mag

    flux = flux - flux.min()
    if flux.max() > 0:
        flux = flux / flux.max()

    # light smoothing so single-bin noise doesn't spike it
    kernel = np.array([0.15, 0.7, 0.15], dtype=np.float32)
    flux = np.convolve(flux, kernel, mode="same")
    flux = np.clip(flux, 0, 1)
    return flux


def hex_or_rgba(color):
    """
    Accept '#RRGGBB', '#RRGGBBAA', or an (r,g,b,a) / (r,g,b) tuple.
    Returns an (r, g, b, a) tuple of ints 0-255.
    """
    if isinstance(color, (tuple, list)):
        if len(color) == 3:
            return (int(color[0]), int(color[1]), int(color[2]), 255)
        return tuple(int(c) for c in color)
    s = color.lstrip("#")
    if len(s) == 6:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return (r, g, b, 255)
    elif len(s) == 8:
        r, g, b, a = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16)
        return (r, g, b, a)
    raise ValueError(f"Unrecognized color format: {color}")
