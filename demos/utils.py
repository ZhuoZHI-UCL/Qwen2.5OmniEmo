# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import soundfile as sf


def ffmpeg_once(in_path: str, out_path: str, fps: float = 5.0, max_pixels: int = 36864):
    """
    统一视频到 fps & 限定像素（保持纵横比、最长边限制）。
    max_pixels=H*W，比如 192*192=36864。
    """
    # 以最长边限制为准：sqrt(max_pixels) 作为最大边
    max_side = int(max_pixels ** 0.5)
    # -vf scale=w:h:force_original_aspect_ratio=decrease
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-r", f"{fps}",
        "-vf", f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease",
        "-an",
        out_path
    ]
    subprocess.run(cmd, check=True)


def extract_audio_wav(in_path: str, wav_path: str, sr: int = 16000):
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vn",
        "-ac", "1",
        "-ar", f"{sr}",
        "-sample_fmt", "s16",
        wav_path
    ]
    subprocess.run(cmd, check=True)


def load_wav_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != target_sr:
        raise RuntimeError(f"Expected {target_sr}Hz, got {sr}. (前面应该已经用 ffmpeg 重采样到 16k)")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav
