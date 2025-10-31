#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import tempfile
import threading
import time
from typing import List

import numpy as np
import torch
from PIL import Image

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

from transformers import AutoModelForConditionalGeneration, AutoProcessor, TextIteratorStreamer

try:
    from qwen_omni_utils import process_mm_info
    HAS_QOU = True
except Exception:
    HAS_QOU = False


def extract_window_frames_with_decord(video_path: str, start_s: float, end_s: float, num_frames: int, tmp_dir: str) -> List[str]:
    """
    从 [start_s, end_s] 等距抽取 num_frames 帧，保存为 JPEG。
    返回 file:// 形式的本地路径列表，便于作为 {"type":"video","video":[...frames...]} 输入。
    """
    if not HAS_DECORD:
        raise RuntimeError("未检测到 decord，请安装 qwen-omni-utils[decord] 或改为 torchvision 方案。")

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    total = len(vr)

    start_idx = max(0, int(math.floor(start_s * fps)))
    end_idx = min(total - 1, int(math.floor(end_s * fps)))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, total - 1)

    idxs = np.linspace(start_idx, end_idx, num_frames, dtype=np.int64)
    frames = vr.get_batch(idxs).asnumpy()  # (N,H,W,3)

    file_uris = []
    for i, arr in enumerate(frames):
        img = Image.fromarray(arr)
        out_path = os.path.join(tmp_dir, f"frame_{start_idx}_{i}.jpg")
        img.save(out_path, quality=90)
        file_uris.append("file://" + os.path.abspath(out_path))
    return file_uris


def build_messages(frame_uris: List[str], question: str, start_s: float, end_s: float) -> List[dict]:
    """
    构建与 Qwen2.5-Omni 兼容的多模态对话消息。
    让模型针对该时间片输出“情绪标签 + 简短原因”。
    """
    system_text = (
        "你是一个面向实时视频的情绪监测助手。你会连续接收视频片段（按时间推进）。"
        "对于每个片段，请简洁输出：当前主要人物的情绪标签（如 happy/sad/angry/surprised/fear/disgust/neutral）"
        "以及不超过15个中文词的简短原因说明。若无法确定，说明不确定及原因。"
    )
    user_text = (
        f"当前片段时间范围: [{start_s:.2f}, {end_s:.2f}] 秒。"
        f"问题：{question}。请简洁回答，不要输出与本片段无关的内容。"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frame_uris},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return messages


def stream_generate(model, processor, messages, max_new_tokens=64, temperature=0.0):
    """
    使用流式生成，返回：
    - first_token_time_abs: 首个 token 到达的绝对时间戳（time.perf_counter）
    - full_text: 全部生成文本
    """
    # 文本模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 多模态预处理（此处不启用视频音轨；如需音频，将 use_audio_in_video=True 并改为分段小视频输入）
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=[text],
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer=processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        return_audio=False,
    )

    first_token_time_abs = None
    chunks = []

    def _gen():
        model.generate(**gen_kwargs)

    th = threading.Thread(target=_gen)
    th.start()
    for token in streamer:
        if first_token_time_abs is None:
            first_token_time_abs = time.perf_counter()
        chunks.append(token)
    th.join()

    return first_token_time_abs, "".join(chunks).strip()


def _read_prompt_from_file(path_prompt: str) -> str:
    """
    从文本文件读取问题（UTF-8），返回去除首尾空白的字符串。
    """
    if not path_prompt:
        raise ValueError("必须提供 path_prompt。")
    if not os.path.isfile(path_prompt):
        raise FileNotFoundError(f"未找到提示文件：{path_prompt}")
    with open(path_prompt, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"提示文件为空：{path_prompt}")
    return content


def main():
    ap = argparse.ArgumentParser(description="Simulate real-time video playback to Qwen2.5-Omni, stream answers and log timestamps/content to JSON.")
    ap.add_argument("--video", type=str, default="/dataset/vid_0001_1.mp4", help="输入视频文件路径")
    ap.add_argument("--out", type=str, default="", help="输出 JSON 文件路径（默认同名 *_realtime_emotions.json）")
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Omni-7B-Instruct", help="Hugging Face model id")
    ap.add_argument("--window-sec", type=float, default=2.0, help="每个时间片窗口长度（秒）")
    ap.add_argument("--hop-sec", type=float, default=0.5, help="窗口步长（秒）")
    ap.add_argument("--frames-per-window", type=int, default=8, help="每个窗口抽取的帧数")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="每次生成的最大新token数")
    ap.add_argument("--temperature", type=float, default=0.0, help="采样温度（0为贪心）")
    ap.add_argument("--attn-impl", type=str, default="flash_attention_2", choices=["flash_attention_2", "eager"], help="注意力实现（GPU 推荐 flash_attention_2）")
    # 由文件提供问题
    ap.add_argument("--path-prompt", type=str, default="scripts/prompt.txt", help="包含问题文本的 UTF-8 编码 txt 文件路径（即 path_prompt）")
    args = ap.parse_args()

    if not HAS_QOU:
        raise RuntimeError("未检测到 qwen-omni-utils，请先安装：pip install qwen-omni-utils[decord] -U")
    if not HAS_DECORD:
        raise RuntimeError("未检测到 decord，请先安装 qwen-omni-utils[decord] -U 或从源码安装 decord。")

    video_path = args.video
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    # 从文件读取问题
    question_text = _read_prompt_from_file(args.path_prompt)

    out_json = args.out or os.path.splitext(os.path.basename(video_path))[0] + "_realtime_emotions.json"

    # 加载模型
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model {args.model_id} ...")
    model = AutoModelForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=(args.attn_impl if torch.cuda.is_available() else "eager"),
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 视频信息
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    video_dur = total_frames / fps if fps > 0 else 0.0
    print(f"Video fps={fps:.3f}, frames={total_frames}, duration={video_dur:.3f}s")

    # 临时帧目录
    tmp_dir = tempfile.mkdtemp(prefix="qwen_omni_frames_")

    # 模拟真实播放的起点
    sim_start = time.perf_counter()

    # 结果容器
    result = {
        "video": os.path.abspath(video_path),
        "model_id": args.model_id,
        "question": question_text,
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "frames_per_window": args.frames_per_window,
        "events": []  # 每次回答的时刻与内容
    }

    # 按窗口推进并实时对齐
    chunk_idx = 0
    cur_start = 0.0
    while cur_start < video_dur - 1e-6:
        cur_end = min(video_dur, cur_start + args.window_sec)

        # 等距抽帧
        frame_uris = extract_window_frames_with_decord(
            video_path=video_path,
            start_s=cur_start,
            end_s=cur_end,
            num_frames=args.frames_per_window,
            tmp_dir=tmp_dir
        )

        # 构造消息（问题来自文件）
        messages = build_messages(frame_uris, question=question_text, start_s=cur_start, end_s=cur_end)

        # 睡眠到该片段播放完成的真实时间点，再触发推理（模拟实时）
        target_wallclock = sim_start + cur_end
        now = time.perf_counter()
        sleep_s = target_wallclock - now
        if sleep_s > 0:
            time.sleep(sleep_s)

        # 流式生成，记录首个token时间
        first_tok_abs, full_text = stream_generate(
            model=model,
            processor=processor,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        first_tok_since_start = (first_tok_abs - sim_start) if first_tok_abs is not None else float("nan")

        # 记录事件
        result["events"].append({
            "chunk_idx": chunk_idx,
            "chunk_start_s": round(cur_start, 3),
            "chunk_end_s": round(cur_end, 3),
            "answer_start_time_since_video_start_s": (round(first_tok_since_start, 3) if isinstance(first_tok_since_start, float) else None),
            "answer_text": full_text
        })

        # 下一窗口
        chunk_idx += 1
        cur_start += args.hop_sec

    # 写出 JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已保存结果到: {os.path.abspath(out_json)}")
    print("完成。")


if __name__ == "__main__":
    main()
