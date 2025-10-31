#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming inference for Qwen2.5-Omni EMO (video+audio), 0.4s per chunk.
严格复用 LLaMA-Factory 的 Route-A 预处理（mm_plugin.routeA_prepare）。
与训练保持一致：在同一条 assistant 消息内续写（add_stream_generation_prompt=False）。
加入丰富的 DEBUG 输出与耗时统计。

用法示例：
CUDA_VISIBLE_DEVICES=0 \
python /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/scripts/routeA_stream_infer_debug.py \
  --model /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora/merged \
  --video /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/LLaMA-Factory/data/dataset/lemon/video/vid_0111_clip16.mp4 \
  --prompt_path /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/user_prompt.txt \
  --print_every 5 --dryrun_first_n 0 --max_new_tokens 64
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Union

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# 避免 tokenizer 多线程刷屏
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- 导入 LLaMA-Factory 的 mm_plugin（Route-A） ----
LFACTORY_SRC = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/LLaMA-Factory/src"
if LFACTORY_SRC not in sys.path:
    sys.path.insert(0, LFACTORY_SRC)
try:
    from llamafactory.data.mm_plugin import routeA_prepare
except Exception as e:
    print(f"[ERROR] Cannot import llamafactory.data.mm_plugin.routeA_prepare: {e}", file=sys.stderr)
    sys.exit(1)


# ------------------------- 小工具函数 -------------------------
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def pretty_size(x: torch.Tensor) -> str:
    if not torch.is_tensor(x):
        return str(type(x))
    numel = x.numel()
    # 近似估计占用
    try:
        dtype_size = torch.finfo(x.dtype).bits // 8 if x.is_floating_point() else torch.iinfo(x.dtype).bits // 8
    except Exception:
        dtype_size = 2
    mb = numel * dtype_size / (1024**2)
    return f"{list(x.shape)} {str(x.dtype)} ~{mb:.2f}MB"

def dump_mm_inputs(mm_inputs: dict, prefix: str = "[MM] "):
    print(prefix + f"keys={list(mm_inputs.keys())}")
    for k, v in mm_inputs.items():
        if torch.is_tensor(v):
            print(prefix + f"{k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            v_show = v if isinstance(v, (int, float, str, list, tuple)) else type(v)
            print(prefix + f"{k}: {v_show}")

def dump_inputs(inputs: dict, prefix: str = "[IN] "):
    keys = list(inputs.keys())
    print(prefix + f"keys={keys}")
    for k in keys:
        v = inputs[k]
        if torch.is_tensor(v):
            print(prefix + f"{k}: {pretty_size(v)}")
        else:
            v_show = v if isinstance(v, (int, float, str, bool)) else type(v)
            print(prefix + f"{k}: {v_show}")

def read_prompt(path: str) -> str:
    p = Path(path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    return text  # 保持与训练一致

def ffmpeg_extract_wav(video_path: str, out_wav: str, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "wav", out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def guess_num_chunks_from_grid(video_grid_thw, second_per_grid: float) -> int:
    if hasattr(video_grid_thw, "tolist"):
        t, h, w = map(int, video_grid_thw[0].tolist())
    else:
        t, h, w = map(int, video_grid_thw[0])
    return int(t)

JSON_OBJ_RE = re.compile(
    r'\{[^{}]*"emotion"\s*:\s*"[^"]+"\s*,[^{}]*"summary_reasoning"\s*:\s*"[^"]+"\s*[^{}]*\}'
)

def extract_json_objects(text: str) -> List[dict]:
    objs = []
    for m in JSON_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if "emotion" in obj and "summary_reasoning" in obj:
                objs.append({"emotion": obj["emotion"], "summary_reasoning": obj["summary_reasoning"]})
        except Exception:
            pass
    return objs

def build_messages(user_prompt: str, n_pairs: int) -> List[dict]:
    t0 = 0.0
    seg = 0.4
    pairs = []
    for _ in range(n_pairs):
        t1 = t0 + seg
        pairs.append(f"<video[{t0:.3f}:{t1:.3f}]><audio[{t0:.3f}:{t1:.3f}]>")
        t0 = t1

    # 模型在每个时间窗口只看最后一个 assistant 的 <video><audio>
    # 输出 emotion + summary_reasoning JSON
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": pairs[-1]},
    ]

def routeA_prepare_safe(
    processor,
    messages: List[dict],
    videos: List[str],
    audio_path: Union[str, None]
) -> Tuple[str, dict, dict]:
    """
    自适应调用 routeA_prepare：
    - 优先 audios=list[str] 形式；
    - 若内部版本差异导致失败，再重试 audios=str。
    与训练一致：add_stream_generation_prompt=False。
    """
    audios_list = [audio_path] if audio_path else []
    t0 = time.perf_counter()
    try:
        out = routeA_prepare(processor, messages, videos, audios_list, add_stream_generation_prompt=False)
        t1 = time.perf_counter()
        print(f"[{now()}][routeA_prepare] audios=list[str] OK, {t1 - t0:.3f}s")
        return out
    except Exception as e1:
        t1 = time.perf_counter()
        print(f"[{now()}][routeA_prepare] list[str] FAILED in {t1 - t0:.3f}s -> retry with str | {type(e1).__name__}: {e1}")
        t2 = time.perf_counter()
        out = routeA_prepare(processor, messages, videos, audio_path, add_stream_generation_prompt=False)
        t3 = time.perf_counter()
        print(f"[{now()}][routeA_prepare] audios=str OK, {t3 - t2:.3f}s")
        return out


# ------------------------- 主逻辑 -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=5, help="每多少个 chunk 打印一次详细 mm_inputs/inputs 形状")
    parser.add_argument("--dryrun_first_n", type=int, default=0, help="仅跑前 N 个 chunk 验证流程，0 表示不裁剪")
    parser.add_argument("--dump_once", action="store_true", help="探针阶段 dump 一次完整的 mm_inputs/inputs")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 0) 加载
    print(f"[{now()}][INFO] Loading processor/model ...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    t1 = time.perf_counter()
    print(f"[{now()}][INFO] Model loaded in {t1 - t0:.3f}s, device={device}, dtype={next(model.parameters()).dtype}")

    # 1) 抽音频
    print(f"[{now()}][INFO] Extracting audio wav ...")
    tmpdir = tempfile.mkdtemp(prefix="emo_stream_")
    wav_path = os.path.join(tmpdir, "audio.wav")
    try:
        ffmpeg_extract_wav(args.video, wav_path, sr=16000)
        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            print(f"[{now()}][WARN] ffmpeg produced empty wav at {wav_path}. Continue without audio.")
            wav_path = None
        else:
            print(f"[{now()}][INFO] WAV ready: {wav_path} ({os.path.getsize(wav_path)/1024:.1f} KB)")
    except Exception as e:
        print(f"[{now()}][WARN] ffmpeg failed, continue without audio. Error: {e}")
        wav_path = None

    user_prompt = read_prompt(args.prompt_path)

    # 2) 探针
    print(f"[{now()}][INFO] Probing shapes via routeA_prepare ...")
    base_messages = build_messages(user_prompt, n_pairs=1)
    videos = [args.video]

    t2 = time.perf_counter()
    try:
        text_probe, mm_inputs_probe, _ = routeA_prepare_safe(processor, base_messages, videos, wav_path)
    except Exception:
        print(f"[{now()}][ERROR] routeA_prepare (probe) failed:\n{traceback.format_exc()}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(2)
    t3 = time.perf_counter()

    video_grid_thw = mm_inputs_probe.get("video_grid_thw", None)
    if video_grid_thw is None:
        print(f"[{now()}][ERROR] video_grid_thw missing from mm_inputs; check av/librosa/soundfile.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(2)
    sec_per_grid = float(mm_inputs_probe.get("video_second_per_grid", [0.4])[0])
    num_chunks = guess_num_chunks_from_grid(video_grid_thw, sec_per_grid)
    print(f"[{now()}][INFO] seconds_per_grid≈{sec_per_grid:.4f}, num_chunks={num_chunks}, probe_time={t3 - t2:.3f}s")

    if args.dump_once:
        print(f"[{now()}] -------- DUMP(once) mm_inputs_probe --------")
        dump_mm_inputs(mm_inputs_probe, prefix="[MM-probe] ")
        print(f"[{now()}] -----------------------------------------")

    # 可选：裁剪
    total_chunks = num_chunks if args.dryrun_first_n <= 0 else min(args.dryrun_first_n, num_chunks)
    if total_chunks != num_chunks:
        print(f"[{now()}][INFO] DRYRUN enabled: only run first {total_chunks}/{num_chunks} chunks")

    # 3) 流式循环
    print(f"[{now()}][INFO] Start streaming inference ...")
    total_t_prepare = 0.0
    total_t_tokenize = 0.0
    total_t_generate = 0.0
    total_t_parse = 0.0

    # 生成停止/截断设置
    stop_strings = ["Human:", "Assistant:", "<|im_end|>", "<|im_start|>"]
    eos_id = processor.tokenizer.eos_token_id

    try:
        for m in range(1, total_chunks + 1):
            ts0 = (m - 1) * sec_per_grid
            ts1 = m * sec_per_grid
            print(f"[{now()}][LOOP] m={m}/{total_chunks}  window=[{ts0:.2f}s–{ts1:.2f}s]")

            # (a) 构造消息（最后一条是 assistant；续写同一条）
            messages_m = build_messages(user_prompt, n_pairs=m)

            # (b) 预处理（含视频/音频展开）
            t_p0 = time.perf_counter()
            text_m, mm_inputs_m, _ = routeA_prepare_safe(processor, messages_m, videos, wav_path)
            t_p1 = time.perf_counter()
            total_t_prepare += (t_p1 - t_p0)
            print(f"[{now()}][STEP] prepare_ok in {t_p1 - t_p0:.3f}s")

            # (c) 打包输入
            t_t0 = time.perf_counter()
            inputs = processor(text=text_m, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            for k, v in mm_inputs_m.items():
                inputs[k] = v.to(device) if torch.is_tensor(v) else v
            inputs["use_audio_in_video"] = getattr(processor, "use_audio_in_video", True)
            t_t1 = time.perf_counter()
            total_t_tokenize += (t_t1 - t_t0)

            # 每 print_every 个 chunk 打印一次形状与关键字段
            if (m % args.print_every == 1) or (m == total_chunks):
                print(f"[{now()}] -------- DUMP inputs/mm_inputs @m={m} --------")
                dump_inputs(inputs, prefix="[IN] ")
                print(f"[{now()}] -------------------------------------------")

            # (d) 生成（与训练一致：续写 assistant，不开新轮）
            t_g0 = time.perf_counter()
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                )
            t_g1 = time.perf_counter()
            total_t_generate += (t_g1 - t_g0)
            print(f"[{now()}][STEP] generate_ok in {t_g1 - t_g0:.3f}s")

            # (e) 解析与截断
            t_s0 = time.perf_counter()
            ctx_len = inputs["input_ids"].size(1)
            gen_ids = gen_out[:, ctx_len:]
            gen_text = processor.batch_decode(
                gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]

            # 先去除已知特殊 token，再基于模板字符串做截断
            gen_text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            for s in stop_strings:
                p = gen_text.find(s)
                if p >= 0:
                    gen_text = gen_text[:p]

            # 判定：是否有 JSON；否则是否是“沉默”（逗号/空白）
            found = extract_json_objects(gen_text)
            if found:
                for obj in found:
                    print(f"[{now()}][OUT] [{ts0:.2f}s–{ts1:.2f}s] {json.dumps(obj, ensure_ascii=False)}")
            else:
                if re.fullmatch(r"[\s,]*", gen_text):
                    print(f"[{now()}][OUT] [{ts0:.2f}s–{ts1:.2f}s] (silence)")
                else:
                    print(f"[{now()}][OUT] [{ts0:.2f}s–{ts1:.2f}s] (raw) {gen_text[:200]}")

            t_s1 = time.perf_counter()
            total_t_parse += (t_s1 - t_s0)

            # 进度心跳
            print(f"[{now()}][STAT] m={m}/{total_chunks}  t_prepare={total_t_prepare:.2f}s  t_tokenize={total_t_tokenize:.2f}s  t_generate={total_t_generate:.2f}s  t_parse={total_t_parse:.2f}s")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n[{now()}][WARN] Interrupted by user. Partial stats below:")

    # 4) 总结
    print(f"[{now()}][INFO] Done. Totals: prepare={total_t_prepare:.2f}s tokenize={total_t_tokenize:.2f}s generate={total_t_generate:.2f}s parse={total_t_parse:.2f}s")
    print(f"[{now()}][INFO] Cleaning temp dir: {tmpdir}")
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