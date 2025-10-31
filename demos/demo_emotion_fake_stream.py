import os, math, json, time, tempfile
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import soundfile as sf
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import gradio as gr

# ---- 环境变量（安静 tokenizers 分叉告警）----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ====== 日志与调试开关 ======
DEBUG = False         # 更细日志（张量设备/形状、prompt 原文等）
LOG_IO = True         # 常规 I/O 日志

def dbg(*a, **k):
    if DEBUG:
        print(*a, **k, flush=True)

def log(*a, **k):
    if LOG_IO:
        print(*a, **k, flush=True)

def log_tensor_dict(name, inputs):
    if not DEBUG:
        return
    print(f"[DEBUG] {name}:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"   {k:<16} device={v.device} dtype={v.dtype} shape={tuple(v.shape)}")
        elif isinstance(v, (list, tuple)):
            print(f"   {k:<16} list/tuple(len={len(v)})")
        else:
            print(f"   {k:<16} type={type(v)}")

# ====== 设备/精度自适应 ======
def prefer_dtype():
    return torch.float16

DTYPE = prefer_dtype()
ATTN_IMPL = "flash_attention_2"  # 可选 "sdpa" | "flash_attention_2"
MODEL_ID = os.environ.get("QWEN_OMNI_CKPT", "Qwen/Qwen2.5-Omni-7B")  # 可换 3B/量化

# ====== 加载模型（文本输出 Thinker，省显存更快） ======
def load_model():
    log(f"[LOAD] MODEL_ID={MODEL_ID}  DTYPE={DTYPE}  ATTN_IMPL={ATTN_IMPL}")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation=ATTN_IMPL,
        device_map=None  # 统一放同一块 GPU，更稳
    ).eval()
    model.to("cuda")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    try:
        first_device = next(model.parameters()).device
        log(f"[LOAD] model first_param_device={first_device}")
    except Exception:
        pass
    return model, processor

MODEL, PROCESSOR = load_model()

# ====== 逐段切片（按 fps -> unit_sec） ======
def iter_video_units(video_clip: VideoFileClip, sr=16000, unit_sec=0.5):
    """
    每次返回 (["<unit>", PIL.Image, np.ndarray_audio_unit], meta)
    meta 包含：t0, t1, t_mid, frame_size, audio_len
    """
    duration = float(video_clip.duration)
    if unit_sec <= 0:
        unit_sec = 0.5
    num_units = max(1, math.ceil(duration / unit_sec))
    for i in range(num_units):
        t0 = i * unit_sec
        t1 = min((i + 1) * unit_sec, duration)
        t_mid = min(t0 + (t1 - t0) / 2.0, duration)

        # 取帧
        frame = video_clip.get_frame(t_mid)
        image = Image.fromarray(frame.astype(np.uint8))
        frame_size = (image.width, image.height)

        # 取子段音频
        audio_seg = video_clip.audio.subclip(t0, t1)
        audio_arr = audio_seg.to_soundarray(fps=sr)  # (N,2) 或 (N,)
        if audio_arr.ndim == 2:
            audio_arr = audio_arr.mean(axis=1)
        audio_np = audio_arr.astype(np.float32)

        if DEBUG and i == 0:
            dbg(f"[DEBUG] first unit audio sample: sr={sr}, len={audio_np.shape[0]}, "
                f"min={audio_np.min():.5f}, max={audio_np.max():.5f}")

        meta = {
            "t0": round(t0, 3),
            "t1": round(t1, 3),
            "t_mid": round(t_mid, 3),
            "frame_size": frame_size,
            "audio_len": int(audio_np.shape[0]),
            "sr": sr
        }
        yield ["<unit>", image, audio_np], meta

# ====== 工具：尝试从半成品 JSON 补齐闭合并解析 ======
def try_salvage_json_from_buffer(cur_json_buf, brace_depth, note=""):
    raw = "".join(cur_json_buf).replace("<|tts_eos|>", "").strip()
    need = max(0, brace_depth)
    raw_fixed = raw + ("}" * need)
    try:
        obj = json.loads(raw_fixed)
        log(f"[JSON-SALVAGE {note}] success; appended_braces={need}")
        return obj
    except Exception as e:
        log(f"[JSON-SALVAGE {note}] fail; appended_braces={need} err={e} raw_fixed={raw_fixed[:300]}{'...[trunc]' if len(raw_fixed)>300 else ''}")
        return None

# ====== 每单元生成并解析一条 JSON ======
def short_json_generate_qwen(conversation, unit_idx=None, max_new_tokens=48):
    """
    使用 Qwen2.5-Omni Thinker 同步生成（贪心），再从文本中抽取第一段 JSON。
    - conversation: Qwen 官方格式
    - 返回 (obj or None, t_first_token模拟时间戳 or None)
    """
    log(f"[GEN-START unit={unit_idx}] generate(max_new_tokens={max_new_tokens}, greedy)")

    # 1) 模板展开
    text = PROCESSOR.apply_chat_template(
        conversation,
        add_generation_prompt=True,  # 在末尾补 assistant 起始
        tokenize=False,
        return_tensors=None
    )
    if DEBUG:
        dbg(f"[DEBUG] chat_template text (first 400): {text[:400]}{'...[trunc]' if len(text)>400 else ''}")

    # 2) 多模态打包
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    if DEBUG:
        dbg(f"[DEBUG] mm_info -> audios={len(audios) if audios else 0}, images={len(images) if images else 0}, videos={len(videos) if videos else 0}")

    # 3) 编码
    try:
        inputs = PROCESSOR(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True
        )
    except TypeError:
        inputs = PROCESSOR(
            text=text,
            audios=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True
        )

    if torch.cuda.is_available():
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    # prompt 长度，后续只解码新 tokens
    prompt_len = inputs["input_ids"].shape[1]
    if DEBUG:
        dbg(f"[DEBUG] prompt_len={prompt_len}")
        log_tensor_dict("inputs(after to cuda)", inputs)

    # 4) 生成 —— 关闭采样，稳定输出 JSON
    with torch.no_grad():
        text_ids = MODEL.generate(
            **inputs,
            do_sample=False,               # 关键：禁采样
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05
        )

    # 只拿新生成部分
    gen_ids = text_ids[:, prompt_len:]
    resp_text = PROCESSOR.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 原样打印模型生成（前 500 字符）
    log(f"[RAW unit={unit_idx}] {resp_text[:500]}{'...[trunc]' if len(resp_text)>500 else ''}")

    # 5) 在响应里搜索第一段 JSON
    in_json = False
    brace_depth = 0
    cur_json_buf = []
    t_first = None
    for ch in resp_text:
        if not in_json:
            if ch == "{":
                in_json = True
                brace_depth = 1
                cur_json_buf = ["{"]
                t_first = time.time()
        else:
            cur_json_buf.append(ch)
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    raw = "".join(cur_json_buf).strip()
                    log(f"[JSON-END unit={unit_idx}] {raw[:300]}{'...[trunc]' if len(raw)>300 else ''}")
                    try:
                        obj = json.loads(raw)
                        # 规范化 changed
                        if "changed" in obj and not isinstance(obj["changed"], bool):
                            if isinstance(obj["changed"], str):
                                obj["changed"] = obj["changed"].strip().lower() == "true"
                            else:
                                obj["changed"] = bool(obj["changed"])
                        return obj, (t_first or time.time())
                    except Exception as e:
                        log(f"[JSON-PARSE-ERR unit={unit_idx}] {e} raw={raw}")
                        return None, None

    # 未闭合则尝试修复
    if in_json and cur_json_buf:
        obj = try_salvage_json_from_buffer(cur_json_buf, brace_depth, note=f"unit={unit_idx}")
        if obj is not None:
            if "changed" in obj and not isinstance(obj["changed"], bool):
                if isinstance(obj["changed"], str):
                    obj["changed"] = obj["changed"].strip().lower() == "true"
                else:
                    obj["changed"] = bool(obj["changed"])
            return obj, (t_first or time.time())

    log(f"[GEN-END unit={unit_idx}] no JSON found")
    return None, None

# ====== 主推理（轮询式）：每单元 -> 送内容 -> 触发问句 -> 生成 -> 解析 ======
def run_stream_inference(video_path, task_prompt, fps=2.0, realtime=True):
    """
    生成器：用于 Gradio 实时刷新事件表（轮询版）
    - fps 控制切片频率（默认 2 FPS，即 unit_sec=0.5）
    - 只在情绪变化时入队；首帧强制入队
    """
    session_id = "sess-" + str(int(time.time() * 1000))
    server_start_ref = time.time()

    fps = float(fps)
    if fps <= 0: fps = 2.0
    unit_sec = 1.0 / fps
    video = VideoFileClip(video_path)
    duration = float(video.duration)
    units_iter = iter_video_units(video, sr=16000, unit_sec=unit_sec)

    log("="*80)
    log(f"[SESSION] id={session_id}  video='{os.path.basename(video_path)}'  duration={duration:.3f}s")
    log(f"[CONFIG] fps={fps:.3f}  unit_sec={unit_sec:.3f}  realtime={realtime}")

    # system prompt 使用官方默认，避免无意义告警
    sys_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    rules = (
        "You will process the video+audio unit-by-unit.\n"
        "When you receive 'CLASSIFY_NOW', respond with EXACTLY ONE JSON object and NOTHING ELSE.\n"
        "Format strictly:\n"
        "{\"emotion\": \"<ONE_ENGLISH_WORD>\", \"reason\": \"<<= 12 words>\", \"changed\": <true|false>} "
        "English only. No code fences. First char must be '{'. Ensure the JSON ends with '}'."
    )

    known_events = []
    last_emotion = None
    start_idx = 0

    for idx, (content, meta) in enumerate(units_iter):
        log("-"*80)
        log(f"[UNIT {idx}] window=[{meta['t0']:.3f}, {meta['t1']:.3f}] mid={meta['t_mid']:.3f}  "
            f"frame={meta['frame_size']}  audio_len={meta['audio_len']}@{meta['sr']}Hz")

        # 拆出图像 & 音频
        _, image, audio_np = content

        # 将当前音频段写入临时 WAV（16kHz, PCM16）
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, audio_np, samplerate=meta["sr"], subtype="PCM_16")
            audio_path = tmp_wav.name

        # 合并“规则 + 触发问句”为同一 user 轮次
        classify_prompt = (
            f"CLASSIFY_NOW. Previous emotion: {last_emotion if last_emotion else 'NONE'}.\n"
            "Return EXACTLY ONE JSON and NOTHING ELSE.\n"
            "Do not output any text before or after the JSON."
        )

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text",  "text": task_prompt + "\n" + rules + "\n\n" + classify_prompt}
                ]
            }
        ]

        # 生成 & 解析 JSON
        obj, t_first = short_json_generate_qwen(
            conversation,
            unit_idx=idx,
            max_new_tokens=48
        )

        # 判定与输出（首帧强制入队）
        if obj and isinstance(obj.get("emotion"), str) and isinstance(obj.get("reason"), str):
            changed_flag = obj.get("changed", None)
            current_emotion = obj["emotion"]

            # ---- 修复点：首帧强制入队，其它帧才参考 changed / 对比 ----
            if last_emotion is None:
                is_changed = True
                dbg(f"[DEBUG] force first event: last_emotion=None -> enqueue")
            else:
                if changed_flag is not None:
                    is_changed = bool(changed_flag)
                    dbg(f"[DEBUG] use model changed flag: {is_changed}")
                else:
                    is_changed = (current_emotion != last_emotion)
                    dbg(f"[DEBUG] local change compare: {current_emotion} vs {last_emotion} -> {is_changed}")

            log(f"[RESULT unit={idx}] obj={obj}  last_emotion={last_emotion}  is_changed={is_changed}")

            if is_changed:
                t_display = max(0.0, (t_first or time.time()) - server_start_ref)
                t_real = float(meta["t1"])
                event = {
                    "t": round(float(t_display), 3),
                    "t_real": round(t_real, 3),
                    "emotion": current_emotion,
                    "reason": obj["reason"]
                }
                sigs = {(e["emotion"], e["t"]) for e in known_events}
                if (event["emotion"], event["t"]) not in sigs:
                    known_events.append(event)
                    log(f"[ENQUEUE unit={idx}] event={event}  (latency ~ t - t_real)")
                    yield list(known_events)
                else:
                    log(f"[DEDUPE unit={idx}] duplicated signature, skip enqueue: {event}")
            else:
                log(f"[SKIP unit={idx}] no change")

            last_emotion = current_emotion
        else:
            if obj is None:
                log(f"[NO-JSON unit={idx}] generation returned None or salvage failed")
            else:
                log(f"[BAD-JSON unit={idx}] missing emotion/reason fields: {obj}")

        # 播放节拍对齐
        if realtime:
            target = server_start_ref + (idx - start_idx + 1) * unit_sec
            to_sleep = target - time.time()
            if to_sleep > 0:
                time.sleep(to_sleep)

        # 清理临时音频
        try:
            os.remove(audio_path)
        except Exception:
            pass

    time.sleep(0.05)
    log("="*80)
    log(f"[SESSION END] total_events={len(known_events)}")

# ====== Prompt（无 t 描述；强调只输出 JSON） ======
TASK_PROMPT = (
    "You continuously watch a streaming video with audio. "
    "Your job is real-time emotion detection for the person in the scene. "
    "Speak only when asked with 'CLASSIFY_NOW'."
)

# ====== Gradio UI ======
with gr.Blocks(title="Qwen2.5-Omni • Emotion Stream (Polling + Robust JSON + Verbose Logs)") as demo:
    gr.Markdown("# Qwen2.5-Omni • Real-time Emotion Change Detection (Polling per Unit)")

    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Video(
                label="Upload a performance video (with audio)",
                sources=["upload"],
                autoplay=False,
                elem_id="video_input"
            )
            fps = gr.Slider(minimum=0.5, maximum=10.0, value=2.0, step=0.5, label="FPS (units per second)")
            realtime = gr.Checkbox(label="Sync with playback time", value=True)
            start_btn = gr.Button("▶️ Start Streaming Analysis", variant="primary")

        with gr.Column(scale=2):
            events = gr.JSON(label="Detected Events (auto-updates)")
            tips = gr.Markdown(
                "- Per unit, the model returns one JSON with current emotion.\n"
                "- We append an event only when the emotion changes (or model sets `changed=true`).\n"
                "- First frame is always enqueued to initialize the timeline.\n"
                "- `t`: time since Start click when the event first token arrives (viewer timeline).\n"
                "- `t_real`: video duration the model has actually seen (up to the end of this unit)."
            )

    def _normalize_video_value(v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return v.get("name") or v.get("path")
        return None

    def on_start(vpath, fps_val, rt):
        vp = _normalize_video_value(vpath)
        if not vp:
            yield []
            return
        yield []  # 清空旧结果
        for item in run_stream_inference(vp, TASK_PROMPT, fps=fps_val, realtime=rt):
            yield item

    start_btn.click(
        fn=on_start,
        inputs=[video, fps, realtime],
        outputs=[events],
        js="""
        (v, fps, rt) => {
            const root = document.querySelector('#video_input');
            if (root) {
                const el = root.querySelector('video');
                if (el) {
                    try {
                        el.pause();
                        el.currentTime = 0;
                        el.play();
                    } catch (e) {}
                }
            }
            return [v, fps, rt];
        }
        """,
        concurrency_limit=1
    )

if __name__ == "__main__":
    demo.queue(max_size=8)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
