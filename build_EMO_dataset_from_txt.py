#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Route-B single-assistant-string dataset for Qwen2.5-Omni (streaming emotion detection).

Changes in this version:
- Use ffprobe to get video/audio/format durations and take the MIN positive as t_end to avoid tail padding.
- Randomly split samples into train/test by 8:2 (configurable via --train_ratio) with a fixed --seed.
- Write two JSONL files under --out_dir:
    built_training.jsonl  (train)
    built_test.jsonl      (test)

Additional changes in THIS revision:
- Support reading user prompt from a text file via --user_prompt_path (UTF-8). If both
  --user_prompt and --user_prompt_path are provided, the file content takes precedence.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _run_ffprobe(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        out = result.stdout.strip()
        return out if out else None
    except Exception:
        return None


def ffprobe_best_duration(path: Path, ffprobe_bin: str = "ffprobe") -> Optional[float]:
    """
    Return the best-guess duration in seconds for timeline slicing.
    Strategy: collect video stream duration, audio stream duration, and format duration;
    pick the MIN positive finite value to avoid trailing padding.
    """
    durations: List[float] = []

    # format duration
    fmt = _run_ffprobe([
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    if fmt:
        try:
            v = float(fmt)
            if math.isfinite(v) and v > 0:
                durations.append(v)
        except Exception:
            pass

    # video stream duration (v:0)
    vdur = _run_ffprobe([
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    if vdur:
        try:
            v = float(vdur)
            if math.isfinite(v) and v > 0:
                durations.append(v)
        except Exception:
            pass

    # audio stream duration (a:0)
    adur = _run_ffprobe([
        ffprobe_bin, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    if adur:
        try:
            v = float(adur)
            if math.isfinite(v) and v > 0:
                durations.append(v)
        except Exception:
            pass

    if not durations:
        return None

    # choose the shortest positive to avoid tail padding
    best = min(durations)
    # round to milliseconds for stable stepping
    return round(best, 3)


def extract_audio_ffmpeg(video_path: Path, audio_path: Path, ffmpeg_bin: str = "ffmpeg") -> None:
    """Extract mono 16kHz WAV from video using ffmpeg (overwrites if exists)."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-vn",               # no video
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        "-sample_fmt", "s16",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)


def ceil_to_next_boundary(ts: float, step: float) -> float:
    """Ceil timestamp to the next chunk boundary: ceil(ts/step)*step (rounded to 3 decimals)."""
    k = math.ceil(ts / step)
    return round(k * step, 3)


def build_routeB_single_assistant_string(
    events: List[Dict[str, Any]],
    t_start: float,
    t_end: float,
    step: float = 0.4
) -> str:
    """
    Build a single assistant content string by slicing the whole [t_start,t_end] timeline into fixed steps.
    - Each chunk contributes <video[t:t+step]><audio[t:t+step]> then either ',' (silence) or a JSON label.
    - Event timestamps are placed at the CEILED boundary (the chunk end) to ensure the model has seen the content.
    """
    # Pre-bucket events by their "speaking boundary"
    buckets: Dict[float, List[Dict[str, Any]]] = {}
    for ev in events:
        ts = float(ev.get("timestamp", 0.0))
        boundary = ceil_to_next_boundary(ts, step)
        # allow t_end equality; anything strictly beyond is dropped
        if boundary > t_end + 1e-6:
            continue
        buckets.setdefault(boundary, []).append(ev)

    # Walk through chunks and build the string
    pieces: List[str] = []
    t = float(t_start)
    while t < t_end - 1e-9:
        t_next = min(t_end, round(t + step, 3))
        pieces.append(f"<video[{t:.3f}:{t_next:.3f}]><audio[{t:.3f}:{t_next:.3f}]>")
        if t_next in buckets:
            for ev in buckets[t_next]:
                label_obj = {
                    "emotion": ev.get("emotion", ""),
                    "summary_reasoning": ev.get("summary_reasoning", ""),
                }
                pieces.append(json.dumps(label_obj, ensure_ascii=False))
        else:
            pieces.append(",")
        t = t_next

    return "".join(pieces)


def load_events_from_txt(txt_path: Path) -> List[Dict[str, Any]]:
    """Each .txt is actually JSON content with a list of events."""
    with open(txt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{txt_path} does not contain a JSON list.")
    # Ensure sorted by timestamp
    data = sorted(data, key=lambda x: float(x.get("timestamp", 0.0)))
    return data


def find_matching_annotation(ann_dir: Path, video_path: Path) -> Optional[Path]:
    """Annotation file name is '<video>.txt' per the user; match by stem (file name without extension)."""
    candidate = ann_dir / f"{video_path.stem}.txt"
    return candidate if candidate.exists() else None


def write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _resolve_user_prompt(args: argparse.Namespace) -> str:
    """
    Resolve user prompt text:
    - If --user_prompt_path is provided, read UTF-8 file content and use it (preferred).
    - Else, use --user_prompt.
    """
    if args.user_prompt_path:
        p = Path(args.user_prompt_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--user_prompt_path not found: {p}")
        if not p.is_file():
            raise IsADirectoryError(f"--user_prompt_path is not a file: {p}")
        with open(p, "r", encoding="utf-8") as f:
            # keep internal newlines but strip a single trailing newline to avoid extra blank line
            text = f.read()
            if text.endswith("\n"):
                text = text[:-1]
        if not text:
            print("[WARN] user prompt file is empty; proceeding with empty prompt.", file=sys.stderr)
        return text
    # fallback to CLI string
    return args.user_prompt


def main():
    parser = argparse.ArgumentParser(description="Build Route-B single-assistant-string dataset for Qwen2.5-Omni.")
    parser.add_argument("--raw_videos", type=str, default='LLaMA-Factory/data/dataset/lemon/video',
                        help="Directory containing raw videos.")
    parser.add_argument("--annotation", type=str, default='LLaMA-Factory/data/dataset/lemon/original_txt_annotations',
                        help="Directory containing per-video .txt annotations.")
    parser.add_argument("--out_dir", type=str, default='LLaMA-Factory/data/dataset/lemon/annotation',
                        help="Output directory. Will write built_training.jsonl and built_test.jsonl here.")
    parser.add_argument("--step", type=float, default=0.4, help="Chunk length in seconds (default: 0.4).")

    # --- Prompt options ---
    parser.add_argument("--user_prompt", type=str, default="Analyze the emotions in the streaming video.",
                        help="User instruction string (ignored if --user_prompt_path is provided).")
    parser.add_argument("--user_prompt_path", type=str, default='/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/user_prompt.txt',
                        help="Path to a UTF-8 text file containing the user instruction. Takes precedence over --user_prompt.")

    parser.add_argument("--extract_audio", type=int, default=1,
                        help="Extract sidecar WAVs with ffmpeg (16kHz mono). Set 0 to disable.")
    parser.add_argument("--audio_dir", type=str, default='LLaMA-Factory/data/dataset/lemon/audio',
                        help="Directory to store extracted audios (if enabled).")
    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg", help="ffmpeg binary path.")
    parser.add_argument("--ffprobe_bin", type=str, default="ffprobe", help="ffprobe binary path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (default: 0.8).")
    parser.add_argument("--max_videos", type=int, default=None, help="Optional limit for debugging.")
    args = parser.parse_args()

    # Resolve prompt (file > CLI string)
    try:
        user_prompt_text = _resolve_user_prompt(args)
    except Exception as e:
        print(f"[ERROR] Failed to load user prompt: {e}", file=sys.stderr)
        sys.exit(1)

    raw_videos = Path(args.raw_videos).resolve()
    ann_dir = Path(args.annotation).resolve()
    out_dir = Path(args.out_dir).resolve()

    # audio dir
    audio_dir = Path(args.audio_dir).resolve() if args.audio_dir else None
    if args.extract_audio:
        assert audio_dir is not None, "--audio_dir must be provided when --extract_audio=1"
        audio_dir.mkdir(parents=True, exist_ok=True)

    # Collect videos
    video_paths: List[Path] = []
    for p in sorted(raw_videos.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            video_paths.append(p)
    if args.max_videos is not None:
        video_paths = video_paths[: args.max_videos]

    samples: List[Dict[str, Any]] = []

    for vid_path in video_paths:
        ann_path = find_matching_annotation(ann_dir, vid_path)
        if not ann_path:
            print(f"[WARN] No annotation for video: {vid_path.name} -> skip.", file=sys.stderr)
            continue

        try:
            events = load_events_from_txt(ann_path)
        except Exception as e:
            print(f"[WARN] Failed to load annotation {ann_path}: {e} -> skip.", file=sys.stderr)
            continue

        # Determine best duration (min of video/audio/format)
        duration = ffprobe_best_duration(vid_path, ffprobe_bin=args.ffprobe_bin)
        if duration is None:
            # Fallback: use last event timestamp rounded up to next boundary + one step buffer
            last_ts = events[-1]["timestamp"] if events else 0.0
            duration = ceil_to_next_boundary(float(last_ts), args.step) + args.step
            print(f"[INFO] ffprobe not available or failed for {vid_path.name}. "
                  f"Fallback duration={duration:.3f}s", file=sys.stderr)

        # Build assistant content
        assistant_str = build_routeB_single_assistant_string(
            events=events,
            t_start=0.0,
            t_end=round(float(duration), 3),
            step=args.step
        )
        meta = {
            "step_seconds": args.step,
            "original_events": events,  # keep original JSON list with timestamps
        }

        # Ensure audio path
        if args.extract_audio:
            audio_path = audio_dir / f"{vid_path.stem}.wav"
            try:
                extract_audio_ffmpeg(vid_path, audio_path, ffmpeg_bin=args.ffmpeg_bin)
            except Exception as e:
                print(f"[WARN] ffmpeg extract failed for {vid_path.name}: {e}. Using empty audio list.", file=sys.stderr)
                audio_list = []
            else:
                audio_list = [str(audio_path.resolve())]
        else:
            audio_list = []

        sample = {
            "messages": [
                {"role": "user", "content": user_prompt_text},
                {"role": "assistant", "content": assistant_str}
            ],
            "videos": [str(vid_path)],
            "audios": audio_list,
            "meta": meta
        }
        samples.append(sample)

    # Split into train/test
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_total = len(samples)
    n_train = int(round(n_total * float(args.train_ratio)))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "built_training.jsonl"
    test_path = out_dir / "built_test.jsonl"

    write_jsonl(train_path, train_samples)
    write_jsonl(test_path, test_samples)

    print(f"[OK] Total samples: {n_total} | Train: {len(train_samples)} | Test: {len(test_samples)}", file=sys.stderr)
    print(f"[OK] Wrote train -> {train_path}", file=sys.stderr)
    print(f"[OK] Wrote test  -> {test_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
