#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案A: 最终可运行版本

完全模仿训练时的数据处理流程
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
import av
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "LLaMA-Factory" / "src"))


# ============================================================================
# 工具函数
# ============================================================================

def build_assistant_content(duration: float, step: float = 0.4) -> str:
    """生成训练格式的assistant content"""
    pieces = []
    t = 0.0
    while t < duration - 1e-9:
        t_next = min(duration, round(t + step, 3))
        pieces.append(f"<video[{t:.3f}:{t_next:.3f}]><audio[{t:.3f}:{t_next:.3f}]>,")
        t = t_next
    return "".join(pieces)


def get_duration(video_path: Path) -> float:
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True
        )
        return round(float(result.stdout.strip()), 3)
    except:
        container = av.open(str(video_path))
        stream = next(s for s in container.streams if s.type == "video")
        dur = float(stream.duration * stream.time_base) if stream.duration else \
              stream.frames / float(stream.average_rate)
        container.close()
        return round(dur, 3)


def load_video(path: Path) -> List[Image.Image]:
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "video")
    frames = [frame.to_image() for frame in container.decode(stream)]
    container.close()
    return frames


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(str(path), sr=sr)
    return audio


# ============================================================================
# 主推理函数
# ============================================================================

def inference(
    video_path: Path,
    audio_path: Path,
    model_path: Path,
    user_prompt: str,
    step: float = 0.4,
    device: str = "cuda",
    max_new_tokens: int = 4096,
) -> List[Dict[str, Any]]:
    
    print("="*80)
    print("开始推理")
    print("="*80)
    
    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    from transformers import (
        AutoTokenizer, 
        AutoProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration
    )
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    print("   ✓ 完成")
    
    # 2. 准备数据
    print("\n[2/6] 准备数据...")
    duration = get_duration(video_path)
    assistant_content = build_assistant_content(duration, step)
    video_frames = load_video(video_path)
    audio_array = load_audio(audio_path)
    
    print(f"   视频: {duration:.3f}s, {len(video_frames)} 帧")
    print(f"   音频: {audio_array.shape}")
    print(f"   Chunks: {assistant_content.count('<video[')}")
    
    # 3. 构建messages
    print("\n[3/6] 构建messages...")
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content}
    ]
    
    # 4. 用plugin处理占位符
    print("\n[4/6] 处理占位符...")
    from llamafactory.data.mm_plugin import get_mm_plugin
    
    plugin = get_mm_plugin(
        name="qwen2_omni",
        image_token="<|IMAGE|>",
        video_token="<|VIDEO|>",
        audio_token="<|AUDIO|>",
        expand_mm_tokens=True
    )
    
    # plugin会替换占位符为实际的token
    processed_msgs = plugin.process_messages(
        messages=messages,
        images=[],
        videos=[video_frames],
        audios=[audio_array],
        processor=processor
    )
    
    print("   ✓ 占位符已替换")
    
    # 5. 生成文本并tokenize
    print("\n[5/6] Tokenize...")
    
    # 应用chat template
    text = tokenizer.apply_chat_template(
        processed_msgs,
        add_generation_prompt=False,
        tokenize=False
    )
    
    print(f"   文本长度: {len(text)} 字符")
    
    # Tokenize (这会得到已经包含<|VIDEO|>和<|AUDIO|> token的序列)
    text_inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    )
    
    # 现在我们需要手动生成multimodal特征
    # 用plugin的_get_mm_inputs方法
    mm_inputs = plugin._get_mm_inputs(
        images=[],
        videos=[video_frames],
        audios=[audio_array],
        processor=processor
    )
    
    # 合并inputs
    inputs = {
        **text_inputs,
        **mm_inputs
    }
    
    # 移到device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items() if v is not None}
    
    print(f"   ✓ input_ids: {inputs['input_ids'].shape}")
    if 'pixel_values_videos' in inputs:
        print(f"   ✓ pixel_values_videos: {inputs['pixel_values_videos'].shape}")
    if 'input_features' in inputs:
        print(f"   ✓ input_features: {inputs['input_features'].shape}")
    
    # 检查token数量
    video_token_id = tokenizer.convert_tokens_to_ids("<|VIDEO|>")
    audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    num_video = (inputs['input_ids'] == video_token_id).sum().item()
    num_audio = (inputs['input_ids'] == audio_token_id).sum().item()
    print(f"   ✓ Video tokens: {num_video}, Audio tokens: {num_audio}")
    
    # 6. Generate
    print("\n[6/6] 生成中...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"   ✓ 生成 {len(generated_ids)} tokens")
    print(f"\n{'='*80}")
    print("生成结果 (前1000字符):")
    print("="*80)
    print(generated_text[:1000])
    if len(generated_text) > 1000:
        print(f"\n... (共 {len(generated_text)} 字符)")
    print("="*80)
    
    # 7. 解析
    print("\n解析结果...")
    pattern = r'\{"emotion":\s*"([^"]+)",\s*"summary_reasoning":\s*"([^"]+)"\}'
    
    results = []
    for match in re.finditer(pattern, generated_text):
        results.append({
            "emotion": match.group(1),
            "summary_reasoning": match.group(2)
        })
    
    if results:
        time_per = duration / len(results)
        for i, r in enumerate(results):
            r["timestamp"] = round((i + 1) * time_per, 3)
    
    print(f"✓ 提取 {len(results)} 个emotions")
    for i, r in enumerate(results[:5]):
        print(f"  [{i+1}] {r['timestamp']:.3f}s: {r['emotion']} - {r['summary_reasoning'][:40]}...")
    
    return results


# ============================================================================
# 命令行
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--model_path", default="output/lemon_omni_lora/merged")
    parser.add_argument("--user_prompt_path", default="user_prompt.txt")
    parser.add_argument("--step", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--output_json", default="results.json")
    args = parser.parse_args()
    
    prompt_file = Path(args.user_prompt_path)
    if prompt_file.exists():
        user_prompt = prompt_file.read_text(encoding="utf-8").strip()
    else:
        user_prompt = "Analyze emotions."
        print(f"[WARN] 使用默认prompt")
    
    print(f"\n配置:")
    print(f"  Video: {args.video_path}")
    print(f"  Audio: {args.audio_path}")
    print(f"  Model: {args.model_path}")
    print(f"  Step: {args.step}s\n")
    
    results = inference(
        video_path=Path(args.video_path),
        audio_path=Path(args.audio_path),
        model_path=Path(args.model_path),
        user_prompt=user_prompt,
        step=args.step,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ 完成! 结果保存到: {args.output_json}")
    print(f"✓ 共提取 {len(results)} 个emotion事件")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
    '''
    CUDA_VISIBLE_DEVICES=0
    python scripts/routeA_stream_infer_debug.py \
    --video_path LLaMA-Factory/data/dataset/lemon/video/vid_0111_clip16.mp4 \
    --audio_path LLaMA-Factory/data/dataset/lemon/audio/vid_0111_clip16.wav \
    --model_path output/lemon_omni_lora_1104/merged \
    --user_prompt_path user_prompt.txt \
    --step 0.4 \
    --max_new_tokens 512 \
    --output_json results_001.json
    '''