#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate chunked streaming inference for Qwen2.5-Omni EMO."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


REPO_ROOT = Path(__file__).resolve().parents[1]
LFACTORY_SRC = str(REPO_ROOT / "LLaMA-Factory" / "src")
if LFACTORY_SRC not in sys.path:
    sys.path.insert(0, LFACTORY_SRC)

from llamafactory.data.mm_plugin import routeA_prepare  # noqa: E402


JSON_OBJ_RE = re.compile(
    r"\{[^{}]*\"emotion\"\s*:\s*\"[^\"]+\"[^{}]*\"summary_reasoning\"\s*:\s*\"[^\"]+\"[^{}]*\}"
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_prompt(path: str) -> str:
    with open(Path(path), "r", encoding="utf-8") as f:
        return f.read()


def ffmpeg_extract_wav(video_path: str, out_wav: str, sr: int = 16000) -> Optional[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        out_wav,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        return None
    return out_wav if Path(out_wav).exists() and Path(out_wav).stat().st_size > 0 else None


def ffprobe_best_duration(path: Path, ffprobe_bin: str = "ffprobe") -> Optional[float]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except Exception:
            return None
        out = res.stdout.strip()
        return out if out else None

    durations: List[float] = []

    fmt = _run(
        [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    if fmt:
        try:
            val = float(fmt)
            if val > 0:
                durations.append(val)
        except Exception:
            pass

    vdur = _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if vdur:
        try:
            val = float(vdur)
            if val > 0:
                durations.append(val)
        except Exception:
            pass

    adur = _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if adur:
        try:
            val = float(adur)
            if val > 0:
                durations.append(val)
        except Exception:
            pass

    if not durations:
        return None

    return round(min(durations), 3)


def build_timeline(duration: float, step: float) -> List[Tuple[float, float]]:
    t = 0.0
    chunks: List[Tuple[float, float]] = []
    while t < duration - 1e-9:
        t_next = min(duration, round(t + step, 3))
        chunks.append((round(t, 3), t_next))
        t = t_next
    return chunks


def extract_json_objects(text: str) -> List[Dict[str, str]]:
    objs = []
    for m in JSON_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if "emotion" in obj and "summary_reasoning" in obj:
            objs.append({"emotion": obj["emotion"], "summary_reasoning": obj["summary_reasoning"]})
    return objs


def cleanup_generation(gen_text: str, stop_strings: List[str]) -> str:
    text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    for stop in stop_strings:
        pos = text.find(stop)
        if pos >= 0:
            text = text[:pos]
    return text.strip()


def take_chunk_token(gen_text: str) -> Tuple[str, Dict[str, Optional[List[Dict[str, str]]]]]:
    stripped = gen_text.lstrip()
    if not stripped:
        return "", {"type": "empty", "events": None}

    json_match = JSON_OBJ_RE.match(stripped)
    if json_match:
        commit = json_match.group(0)
        return commit, {"type": "json", "events": extract_json_objects(commit)}

    if stripped[0] == ",":
        return ",", {"type": "silence", "events": None}

    return stripped, {"type": "raw", "events": None}


def routeA_prepare_safe(processor, messages, videos, audio_path):
    audios = [audio_path] if audio_path else []
    try:
        return routeA_prepare(processor, messages, videos, audios, add_stream_generation_prompt=False)
    except Exception:
        return routeA_prepare(processor, messages, videos, audio_path, add_stream_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate chunked streaming inference for Qwen2.5-Omni EMO")
    parser.add_argument("--model", type=str, required=True, help="Path to the merged Qwen2.5-Omni EMO model")
    parser.add_argument("--video", type=str, required=True, help="Video file to analyse")
    parser.add_argument("--prompt_path", type=str, required=True, help="Prompt text used during training")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g., cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dryrun_chunks", type=int, default=0, help="Limit to first N chunks (0 = all)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"[{now()}][INFO] Loading processor and model ...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    step_seconds = float(getattr(processor, "seconds_per_chunk", 0.4))

    tmpdir = tempfile.mkdtemp(prefix="emo_stream_")
    wav_path = ffmpeg_extract_wav(args.video, os.path.join(tmpdir, "audio.wav"))
    if wav_path:
        print(f"[{now()}][INFO] Audio extracted -> {wav_path}")
    else:
        print(f"[{now()}][WARN] Audio extraction failed, continue without audio")

    user_prompt = read_prompt(args.prompt_path)

    duration = ffprobe_best_duration(Path(args.video))
    if duration is None:
        raise RuntimeError("Failed to measure video duration via ffprobe")

    timeline = build_timeline(duration, step_seconds)
    if args.dryrun_chunks > 0:
        timeline = timeline[: args.dryrun_chunks]

    videos = [args.video]
    stop_strings = ["Human:", "Assistant:", "<|im_start|>", "<|im_end|>"]

    assistant_text = ""
    committed: List[Dict[str, object]] = []

    for idx, (t0, t1) in enumerate(timeline, start=1):
        window = f"[{t0:.2f}s–{t1:.2f}s]"
        print(f"[{now()}][LOOP] chunk={idx}/{len(timeline)} window={window}")

        placeholder = f"<video[{t0:.3f}:{t1:.3f}]><audio[{t0:.3f}:{t1:.3f}]>"
        assistant_text += placeholder
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]

        text, mm_inputs, _ = routeA_prepare_safe(processor, messages, videos, wav_path)

        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        for k, v in mm_inputs.items():
            inputs[k] = v.to(device) if torch.is_tensor(v) else v
        inputs["use_audio_in_video"] = getattr(processor, "use_audio_in_video", True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        ctx = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, ctx:]
        gen_text = processor.batch_decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        cleaned = cleanup_generation(gen_text, stop_strings)
        commit_text, meta = take_chunk_token(cleaned)
        if not commit_text:
            commit_text = ","
            meta = {"type": "silence", "events": None, "note": "forced"}

        assistant_text += commit_text

        entry: Dict[str, object] = {
            "start": t0,
            "end": t1,
            "text": commit_text,
            "meta": meta,
        }
        committed.append(entry)

        if meta["type"] == "json":
            for event in meta.get("events") or []:
                print(f"[{now()}][OUT] {window} {json.dumps(event, ensure_ascii=False)}")
        elif meta["type"] == "silence":
            print(f"[{now()}][OUT] {window} (silence)")
        else:
            print(f"[{now()}][OUT] {window} (raw) {commit_text[:200]}")

    print(f"[{now()}][INFO] Streaming finished: {len(committed)} chunks")
    summary = [c for c in committed if c["meta"]["type"] == "json"]
    if summary:
        print(f"[{now()}][INFO] Detected {len(summary)} emotion changes:")
        for item in summary:
            for ev in item["meta"].get("events") or []:
                print(f"  - [{item['start']:.2f}s–{item['end']:.2f}s] {ev['emotion']}: {ev['summary_reasoning']}")
    else:
        print(f"[{now()}][INFO] No emotion changes detected")

    print(f"[{now()}][INFO] Cleaning temporary files -> {tmpdir}")
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=0 \
python /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/scripts/routeA_stream_infer_debug.py \
  --model /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora/merged \
  --video /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/LLaMA-Factory/data/dataset/lemon/video/vid_0111_clip16.mp4 \
  --prompt_path /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/user_prompt.txt
'''