# -*- coding: utf-8 -*-
import os
import re
import json
import math
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import torch
import gradio as gr
import transformers
from decord import VideoReader, cpu

logger = transformers.logging.get_logger("qwen_emo_demo")

# ---- 关掉代理，避免 SSL/HF 访问问题 ----
for k in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = ",".join(set(filter(None, [os.environ.get("NO_PROXY",""), "127.0.0.1","localhost","::1"])))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

# ========= 需要确认/可修改的路径 =========
FINETUNED_BASE = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora/merged"
USER_PROMPT_PATH = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/user_prompt.txt"
ANNOT_ROOT = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/LLaMA-Factory/data/dataset/lemon/original_txt_annotations"

# ========= 与训练保持一致的常量 =========
SECONDS_PER_CHUNK_DEFAULT = 0.4
POSITION_ID_PER_SECONDS_DEFAULT = 25
COMMA_TOKEN_ID_DEFAULT = 11

# ========= 构建模型/处理器 =========
def load_model_and_processor():
    from transformers import (
        AutoProcessor,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
    logger.warning("Loading model & processor ...")
    try:
        processor = Qwen2_5OmniProcessor.from_pretrained(
            FINETUNED_BASE, trust_remote_code=True
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            FINETUNED_BASE, trust_remote_code=True
        )
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        FINETUNED_BASE,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor

# ========= 状态 =========
@dataclass
class StreamState:
    past_key_values: Optional[Tuple] = None
    last_ids: Optional[torch.LongTensor] = None
    model_seen_chunks: int = 0
    video_frames_thwc: Optional[np.ndarray] = None
    src_fps: float = 0.0
    audio_wav: Optional[np.ndarray] = None
    audio_sr: int = 16000

# ========= 推理器 =========
class LiveInferQwen:
    def __init__(self):
        self.model, self.processor = load_model_and_processor()
        self.device = next(self.model.parameters()).device

        cfg = self.model.config
        tok = getattr(self.processor, "tokenizer", None)

        self.hidden_size = int(self.model.get_input_embeddings().embedding_dim)
        self.seconds_per_chunk = (
            getattr(cfg, "seconds_per_chunk", None)
            or getattr(self.processor, "seconds_per_chunk", None)
            or SECONDS_PER_CHUNK_DEFAULT
        )
        self.position_id_per_seconds = (
            getattr(cfg, "position_id_per_seconds", None)
            or getattr(self.processor, "position_id_per_seconds", None)
            or POSITION_ID_PER_SECONDS_DEFAULT
        )
        self.chunk_ntokens = int(self.position_id_per_seconds * self.seconds_per_chunk)

        self.eos_token_id = (
            getattr(cfg, "eos_token_id", None)
            or (tok.eos_token_id if tok is not None else None)
            or 151643
        )
        self.comma_id = getattr(cfg, "stream_comma_token_id", None) or COMMA_TOKEN_ID_DEFAULT
        self.audio_eos_id = getattr(cfg, "audio_end_token_id", getattr(cfg, "audio_eos_token_id", 151648))

        self.force_after = 6
        self.silent_counter = 0

        if os.path.exists(USER_PROMPT_PATH):
            with open(USER_PROMPT_PATH, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        else:
            system_prompt = "You are an online video emotion assistant. Output emotion label and reasoning."
        self.system_prompt = system_prompt

        self._start_ids = self.processor.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': self.system_prompt}],
            add_generation_prompt=False,
            return_tensors='pt'
        ).to(self.device)

        self.state = StreamState()

    def reset(self):
        self.state = StreamState()
        self.silent_counter = 0

    # ---------- 读取整段视频（帧） ----------
    def _load_video_frames(self, video_path: str) -> Tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(range(len(vr))).asnumpy()
        try:
            src_fps = float(vr.get_avg_fps())
        except Exception:
            src_fps = len(vr) / max(1.0, (len(vr)/5.0/5.0))
        return frames, src_fps

    # ---------- ffmpeg 提取整段音频 ----------
    def _extract_wav_via_ffmpeg(self, video_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        tmp_wav = "/tmp/qwen_omni_tmp.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(target_sr), "-f", "wav", tmp_wav
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            wav, sr = sf.read(tmp_wav)
            if wav.ndim == 2:
                wav = np.mean(wav, axis=1)
            return wav.astype(np.float32), sr
        except Exception as e:
            logger.warning(f"[audio] read ffmpeg wav failed: {e}")
            return np.zeros(target_sr, dtype=np.float32), target_sr

    # ---------- 对外：载入视频（整段） ----------
    def load_video(self, video_path: str):
        frames, src_fps = self._load_video_frames(video_path)
        wav, sr = self._extract_wav_via_ffmpeg(video_path, target_sr=16000)
        self.state.video_frames_thwc = frames
        self.state.src_fps = src_fps
        self.state.audio_wav = wav
        self.state.audio_sr = sr
        self.state.model_seen_chunks = 0
        self.state.past_key_values = None
        self.state.last_ids = None
        logger.warning(f"video loaded: frames={frames.shape}, src_fps={src_fps:.3f}; audio={wav.shape} sr={sr}")

    # ---------- 截取一个 [t0,t1) 子片段 ----------
    def _slice_clip_for_chunk(self, t0: float, t1: float):
        frames = self.state.video_frames_thwc
        src_fps = self.state.src_fps
        T = frames.shape[0]

        s_idx = int(max(0, math.floor(t0 * src_fps)))
        e_idx = int(min(T, max(s_idx + 1, math.floor(t1 * src_fps))))
        clip_v = frames[s_idx:e_idx]
        if clip_v.size == 0:
            s_idx = min(T - 1, s_idx)
            e_idx = min(T, s_idx + 1)
            clip_v = frames[s_idx:e_idx]

        wav = self.state.audio_wav
        sr = self.state.audio_sr
        a0 = int(max(0, math.floor(t0 * sr)))
        a1 = int(min(len(wav), max(a0 + 1, math.floor(t1 * sr))))
        clip_a = wav[a0:a1]
        if clip_a.size == 0:
            clip_a = np.zeros(int((t1 - t0) * sr) or 160, dtype=np.float32)

        print(f"[slice] t=[{t0:.2f},{t1:.2f})  v_idx=[{s_idx},{e_idx})  a_idx=[{a0},{a1})  "
              f"v_shape={clip_v.shape}  a_len={clip_a.shape[0]}", flush=True)
        return clip_v, (clip_a, sr)

    # ---------- 构建单个 chunk 的 processor 输入 ----------
    def _build_chunk_inputs(self, t0: float, t1: float):
        user_msg = f"<video[{t0:.3f}:{t1:.3f}]><audio[{t0:.3f}:{t1:.3f}]>"
        conv = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_msg},
        ]
        text = self.processor.apply_chat_template(conv, tokenize=False)

        clip_v, clip_a = self._slice_clip_for_chunk(t0, t1)
        inputs = self.processor(
            text=text,
            videos=[clip_v],
            audios=[clip_a],
            padding=True,
            return_tensors="pt",
        )
        for k in list(inputs.keys()):
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(self.device)

        input_ids = inputs["input_ids"]
        cfg = self.model.config
        vtok = (input_ids == getattr(cfg, "video_token_id", 151656)).sum().item()
        atok = (input_ids == getattr(cfg, "audio_token_id", 151646)).sum().item()
        print(f"[placeholder] n_video_tokens={vtok}  n_audio_tokens={atok}", flush=True)
        return inputs

    # ---------- 执行一个 chunk 前传 + 门控/生成 ----------
    def step_chunk(self, real_time_s: float) -> Optional[str]:
        target_idx = int(real_time_s / self.seconds_per_chunk)
        if target_idx <= self.state.model_seen_chunks:
            return None

        ret_text = None
        while self.state.model_seen_chunks < target_idx:
            k = self.state.model_seen_chunks
            t0 = k * self.seconds_per_chunk
            t1 = (k + 1) * self.seconds_per_chunk

            inputs = self._build_chunk_inputs(t0, t1)

            if self.state.past_key_values is None:
                self.state.last_ids = self._start_ids

            step_ids = torch.cat([self.state.last_ids, inputs["input_ids"]], dim=1)
            inputs_embeds = self.model.get_input_embeddings()(step_ids)

            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(step_ids, dtype=torch.long, device=step_ids.device),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                video_grid_thw=inputs.get("video_grid_thw"),
                input_features=inputs.get("input_features"),
                feature_attention_mask=inputs.get("feature_attention_mask"),
                use_audio_in_video=False,
                use_cache=True,
                past_key_values=self.state.past_key_values,
            )
            self.state.past_key_values = outputs.past_key_values

            next_id = outputs.logits[:, -1, :].argmax(dim=-1).view(1, 1)
            print(f"[stream] chunk={k} t1={t1:.2f}s next_id={next_id.item()} (comma_id={self.comma_id})", flush=True)

            if next_id.item() == self.comma_id:
                self.state.last_ids = next_id
                self.silent_counter += 1
                ret = None
                if self.silent_counter >= self.force_after:
                    print(f"[force] after {self.silent_counter} silent chunks → force generate", flush=True)
                    ret = self._generate_response(prefix_id=torch.tensor([[self.comma_id+1]], device=next_id.device))
                    self.silent_counter = 0
            else:
                self.silent_counter = 0
                ret = self._generate_response(prefix_id=next_id, max_new_tokens=96)

            self.state.model_seen_chunks += 1
            if ret:
                ret_text = ret

        return ret_text

    # ---------- 贪心生成 ----------
    def _generate_response(self, prefix_id: torch.LongTensor, max_new_tokens: int = 64) -> str:
        generated = [prefix_id]
        cur_ids = prefix_id
        for _ in range(max_new_tokens - 1):
            inputs_embeds = self.model.get_input_embeddings()(cur_ids)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=self.state.past_key_values,
            )
            self.state.past_key_values = outputs.past_key_values
            next_id = outputs.logits[:, -1, :].argmax(dim=-1).view(1, 1)
            generated.append(next_id)
            cur_ids = next_id
            if next_id.item() == self.eos_token_id:
                break

        gen_ids = torch.cat(generated, dim=1)
        if hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode(
                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
        else:
            text = self.processor.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
        text = text.strip()
        print(f"[gen] >>> {text}", flush=True)
        return text

# ========= 注释读取 =========
def load_annotation_for_video(src_video_path: str) -> str:
    base = os.path.basename(src_video_path)
    stem, _ = os.path.splitext(base)
    txt_path = os.path.join(ANNOT_ROOT, f"{stem}.txt")
    if not os.path.exists(txt_path):
        return f"(No annotation found for {stem}.txt)"
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lines = []
        for item in data:
            ts = item.get("timestamp", "-")
            emo = item.get("emotion", "-")
            sr = item.get("summary_reasoning", "")
            dr = item.get("detailed_reasoning", "")
            lines.append(f"[{ts:>6}]  {emo}\n  - {sr}\n  - {dr}\n")
        return "### Annotations\n\n" + "\n".join(lines)
    except Exception as e:
        return f"(Fail to parse annotation: {e})"

# ========= Gradio UI =========
CSS = """
#gr_title {text-align:center;}
#left_col {max-width: 720px;}
#right_col {max-height: 640px; overflow:auto;}
#gr_video {max-height: 480px;}
"""

# JS：Start 时让视频播放，并翻转 gate
START_JS = """() => {
  const v = document.querySelector("#gr_video video");
  try {
    if (v) { v.currentTime = 0; v.play(); }
  } catch (e) { /* 某些浏览器需用户手势，忽略错误 */ }
  return 1; // tick_gate 设为 1，触发轮询
}"""

# JS：轮询时读取 video.currentTime，把它传给后端的 poll_and_step
# 输入为 (history, video, gate)；输出给 python 函数的是 (history, currentTime, gate)
TICK_JS = """(history, _video, gate) => {
  const v = document.querySelector("#gr_video video");
  const t = v ? v.currentTime : 0.0;
  return [history, t, gate];
}"""

live = LiveInferQwen()

with gr.Blocks(title="Qwen2.5-Omni-EMO Online Demo", css=CSS) as demo:
    gr.Markdown("# Qwen2.5-Omni-EMO Online Emotion Demo", elem_id="gr_title")
    with gr.Row():
        with gr.Column(elem_id="left_col"):
            video = gr.Video(
                label="Upload Video (click play to start)",
                elem_id="gr_video",
                autoplay=False,          # 我们用 JS 控制 play()
                sources=["upload"],
            )
            btn_start = gr.Button("▶️ Start / Reset")
            time_display = gr.Markdown("**Real time:** 0.0s &nbsp;&nbsp; **Model seen:** 0.0s")

            # 轮询门（flip 触发）
            tick_gate = gr.Number(value=0, visible=False)

        with gr.Column(elem_id="right_col"):
            md_annotation = gr.Markdown("(Annotation will appear here)")
            # 使用 messages 模式
            chat = gr.Chatbot(label="Streaming Outputs", type="messages")

    # 上传视频 → 重置状态，加载视频，显示注释，清空历史
    def on_video_change(src_path):
        if not src_path:
            return gr.update(value="(No video)"), []
        live.reset()
        live.load_video(src_path)
        ann = load_annotation_for_video(src_path)
        return ann, []

    video.change(on_video_change, inputs=[video], outputs=[md_annotation, chat])

    # 点击开始/重置：重置后端状态 + 前端播放视频 + 翻转 gate
    def on_start_py():
        live.reset()
        # 这里只返回一个占位，实际 gate 的值由前端 JS 决定（返回 1）
        return 0

    btn_start.click(
        on_start_py,
        inputs=None,
        outputs=[tick_gate],
        js=START_JS
    )

    # 轮询：先在 JS 里读 currentTime → 再把 (history, cur_t, gate) 交给后端
    def poll_and_step(history, cur_t, gate):
        try:
            print(f"[poll] cur_t={float(cur_t):.3f}, gate={gate}", flush=True)
        except Exception:
            print("[poll] cur_t=?, gate=?", flush=True)

        resp = live.step_chunk(real_time_s=float(cur_t))

        real_t = float(cur_t)
        model_seen = live.state.model_seen_chunks * live.seconds_per_chunk
        time_md = f"**Real time:** {real_t:.1f}s &nbsp;&nbsp; **Model seen:** {model_seen:.1f}s"

        if resp:
            msg = f"(Real={real_t:.1f}s | Seen={model_seen:.1f}s) {resp}"
            history = history or []
            history.append({"role": "assistant", "content": msg})

        # gate 翻转以继续触发
        return history, time_md, (gate + 1) % 2

    tick_gate.change(
        poll_and_step,
        # 注意：把 video 作为输入之一，使 JS 有机会访问它并返回 currentTime
        inputs=[chat, video, tick_gate],
        outputs=[chat, time_display, tick_gate],
        js=TICK_JS,
        queue=True
    )

    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
