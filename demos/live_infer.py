# -*- coding: utf-8 -*-
from .utils import ffmpeg_once, extract_audio_wav, load_wav_16k
import logging, math, os, json, time, torch, collections, numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torchvision
from torchvision.io import read_video
logger = logging.getLogger("liveinfer")
logging.basicConfig(level=logging.INFO)
# === 关键：本地类 ===
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor,
)
from transformers import AutoConfig, AutoModelForCausalLM

@dataclass
class InferConfig:
    model_path: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16  # 你训练时是 bf16
    # === 与训练保持一致 ===
    seconds_per_chunk: float = 0.4
    video_fps: float = 5.0
    video_max_pixels: int = 192 * 192
    video_maxlen: int = 300
    audio_sr: int = 16000
    # === 推理门控（沉默 token = ','）===
    comma_token_id: int = 11          # 你在loss里也是这样取
    comma_silence_threshold: float = 0.72  # 当','概率 >= 阈值 → 认为沉默（可按需微调）
    # === System prompt（与你训练一致/兼容）===
    system_prompt: str = (
        "You are a multimodal emotion analysis assistant.\n"
        "You will receive continuous video and audio streams and must identify the dominant emotion of a person, "
        "outputting a new JSON object only when an emotion change is detected.\n"
        "### Instructions:\n"
        "1. Multimodal Reasoning: Pay attention to visual cues (facial expressions, body language, scene changes) "
        "and audio cues (tone, pitch, rhythm).\n"
        "2. Emotion Labeling: When you detect an emotion change, output a simple, clear, single-word label "
        "describing the emotion (e.g., “happy,” “angry,” “sad,” “calm,” “nervous”). Avoid complex or poetic words.\n"
        "3. Reasoning Output: For each detected change, provide:\n"
        "   - The emotion label.\n"
        "   - A summary_reasoning: only the most important cues, expressed in the shortest form possible "
        "(no extra words, no connectors).\n"
        "### Output Format:\n"
        "Return the results in JSON format with the following keys for each detected emotion change:\n"
        '- "emotion": <string, one word>\n'
        '- "summary_reasoning": <string>\n'
        "### Example Output:\n"
        '[{\"emotion\":\"calm\",\"summary_reasoning\":\"Steady voice, neutral face.\"}]'
    )


class LiveInfer:
    """
    整段视频/音频先做**一次**预处理（与训练保持一致），
    然后构造“每个 0.4s chunk 对应的 (n_video_tokens, n_audio_tokens)”索引，
    流式播放时，每到一个 chunk 就把对应数量的 video/audio 连续 token 嵌入拼接进 KV-cache，
    只要当前步的 argmax 是非逗号（或逗号置信<阈值），就触发“说话”，
    进行一次短的 greedy 生成，得到一条 JSON 片段并追加到输出。
    """
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg

        # === A. 先注册 AutoConfig / AutoModelForCausalLM 的映射（关键！）===
        # 使 Auto 能识别 model_type="qwen2_5_omni_thinker"
        try:
            AutoConfig.register("qwen2_5_omni_thinker", Qwen2_5OmniThinkerConfig)
        except Exception:
            # 重复注册会抛异常，这里忽略即可
            pass

        try:
            AutoModelForCausalLM.register(Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration)
        except Exception:
            pass

        # 设备 / dtype
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # === B. 加载 Processor / Model / Config（全部 trust_remote_code=False）===
        self.processor: Qwen2_5OmniProcessor = Qwen2_5OmniProcessor.from_pretrained(
            cfg.model_path, trust_remote_code=False
        )

        self.model: Qwen2_5OmniThinkerForConditionalGeneration = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            cfg.model_path,
            torch_dtype=cfg.dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=False,   # 必须 False
        ).to(self.device)
        self.model.eval()

        model_cfg = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=False)
        sec_conf = getattr(getattr(model_cfg, "processor_config", model_cfg), "seconds_per_chunk", None)
        if sec_conf is None:
            sec_conf = getattr(self.processor, "seconds_per_chunk", None)
        logger.info(f"[Check] seconds_per_chunk (model/proc) = {sec_conf}, expect = {cfg.seconds_per_chunk}")
        # 不强制报错，但建议一致
        self.pos_per_sec = getattr(self.processor, "position_id_per_seconds", 25)
        self.tokens_per_chunk = int(round(self.pos_per_sec * cfg.seconds_per_chunk))
        logger.info(f"[Check] tokens_per_chunk (pos/s * sec) = {self.tokens_per_chunk}")

        # 3) System prompt → 起始 ids
# ---- 统一生成 system input_ids（兼容返回 Tensor 或 Dict）----
        sys_msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": cfg.system_prompt}
                ],
            }
        ]
        sys_ret = self.processor.apply_chat_template(
            sys_msgs,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
        )

        # 兼容不同实现：有的返回 dict({'input_ids': ...})，有的直接返回 Tensor
        if isinstance(sys_ret, dict):
            sys_input_ids = sys_ret["input_ids"]
        else:
            sys_input_ids = sys_ret  # 已是 Tensor

        self._system_input_ids = sys_input_ids.to(self.device)

        # 4) 状态
        self.reset()

    # ---------- 生命周期控制 ----------
    def reset(self):
        self.video_tensor = None           # T,C,H,W
        self.audio_wave = None             # 1D numpy
        self.video_grid_thw = None
        self.video_embeds_flat = None      # [Nv, D]
        self.audio_embeds_flat = None      # [Na, D]
        self.v_chunks: List[Tuple[int, int]] = []  # [(s,e), ...] in video token index
        self.a_chunks: List[Tuple[int, int]] = []  # [(s,e), ...] in audio token index
        self.v_ptr = 0
        self.a_ptr = 0
        self.time_ptr = 0.0
        self.past_key_values = None
        self.last_input_ids = None
        self.hidden_size = self.model.config.text_config.hidden_size
        self.generated: List[str] = []     # 累积的 JSON 片段（字符串）
        torch.cuda.empty_cache()

    # ---------- 预处理 ----------
    def _video_preprocess(self, raw_video_path: str) -> str:
        """
        用 ffmpeg 统一转成 cfg.video_fps 和合适分辨率；返回缓存路径。
        """
        name, ext = os.path.splitext(os.path.basename(raw_video_path))
        cache_path = os.path.join(
            os.path.dirname(raw_video_path),
            f"{name}_{int(self.cfg.video_fps)}fps_{int(math.sqrt(self.cfg.video_max_pixels))}p{ext}"
        )
        if not os.path.exists(cache_path):
            ffmpeg_once(
                in_path=raw_video_path,
                out_path=cache_path,
                fps=self.cfg.video_fps,
                max_pixels=self.cfg.video_max_pixels
            )
            logger.info(f"[FFmpeg] wrote {cache_path}")
        return cache_path

    def _load_whole_video_tensor(self, path: str):
        # read_video 使用 pts_unit='sec' 可获取稳定帧序列
        video, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
        # 规范到 (T,C,H,W)
        if video.dtype != torch.uint8:
            video = (video.clamp(0, 1) * 255).byte()
        self.video_tensor = video.to(self.device, non_blocking=True)
        logger.info(f"[Video] tensor={tuple(self.video_tensor.shape)} @ {self.cfg.video_fps} FPS")

    def _extract_and_load_audio(self, video_path: str) -> Optional[str]:
        wav_path = os.path.splitext(video_path)[0] + ".wav"
        if not os.path.exists(wav_path):
            extract_audio_wav(video_path, wav_path, sr=self.cfg.audio_sr)
        wav = load_wav_16k(wav_path, target_sr=self.cfg.audio_sr)  # np.float32 [-1,1]
        self.audio_wave = wav
        logger.info(f"[Audio] wav={wav.shape}, sr={self.cfg.audio_sr}")
        return wav_path

    def _precompute_all_features(self):
        """
        与你训练时的 processor 路径一致：
        - video_processor: 产生 pixel_values_videos + video_grid_thw
        - whisper feature_extractor: 产生 input_features + feature_attention_mask
        之后用模型 encoder 得到 video/audio 连续嵌入序列，再构建“chunk 索引”。
        """
        # 1) Video
        # 走 omni 的 video_processor（和你训练时的 plugin _get_mm_inputs 内一致）
        video_dict = dict(videos=[self.video_tensor], images=None)
        vp = getattr(self.processor, "video_processor")
        out_v = vp(videos=video_dict["videos"], return_tensors="pt")
        pixel_values_videos = out_v["pixel_values_videos"].to(self.device, dtype=self.model.dtype)
        self.video_grid_thw = out_v["video_grid_thw"].to(self.device)
        temporal_patch_size = getattr(vp, "temporal_patch_size", 2)  # 你训练时是 2
        video_second_per_grid = torch.tensor(
            [temporal_patch_size / self.cfg.video_fps], device=self.device, dtype=torch.float32
        )
        # 调模型视觉塔拿特征（与你 forward() 相同逻辑）
        with torch.no_grad():
            video_embeds = self.model.get_video_features(pixel_values_videos, self.video_grid_thw)  # [Nv, D]
        self.video_embeds_flat = video_embeds  # 已经是扁平 token 序列
        logger.info(f"[Video] embeds={tuple(self.video_embeds_flat.shape)}")

        # 2) Audio（Whisper feature_extractor → model.get_audio_features）
        feat_extractor = getattr(self.processor, "feature_extractor")
        audio_inputs = feat_extractor(
            [self.audio_wave], sampling_rate=self.cfg.audio_sr,
            return_tensors="pt", return_attention_mask=True, padding="max_length"
        )
        input_features = audio_inputs["input_features"].to(self.device, dtype=self.model.dtype)
        feature_attention_mask = audio_inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            audio_embeds = self.model.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask
            )  # [Na, D]
        self.audio_embeds_flat = audio_embeds
        logger.info(f"[Audio] embeds={tuple(self.audio_embeds_flat.shape)}")

        # 3) 构建 chunk 索引（严格照你训练时 process_messages 的切法）
        #    - Video: 先用 grid 时间索引 → 展开成 token 粒度 → processor.get_chunked_index
        T, H, W = list(map(int, self.video_grid_thw[0].tolist()))
        merge = getattr(vp, "merge_size", 2)
        v_tpg = (H // merge) * (W // merge)  # 每个 temporal grid 有多少 video tokens
        sec_per_grid = float(video_second_per_grid[0])
        t_index = (torch.arange(T, device=self.device) * sec_per_grid * self.pos_per_sec).long()
        video_t_index = t_index.view(-1, 1).expand(-1, v_tpg).reshape(-1)  # [Nv]
        v_chunks = self.processor.get_chunked_index(video_t_index, self.tokens_per_chunk)
        self.v_chunks = [(int(s), int(e)) for s, e in v_chunks if int(e) > int(s)]

        #    - Audio: 25Hz 均匀时间索引 → get_chunked_index
        Na = self.audio_embeds_flat.size(0)
        audio_t_index = torch.arange(Na, device=self.device).long()
        a_chunks = self.processor.get_chunked_index(audio_t_index, self.tokens_per_chunk)
        self.a_chunks = [(int(s), int(e)) for s, e in a_chunks if int(e) > int(s)]

        # 截断到最短（避免 chunk 数不同步）
        n_pairs = min(len(self.v_chunks), len(self.a_chunks))
        self.v_chunks = self.v_chunks[:n_pairs]
        self.a_chunks = self.a_chunks[:n_pairs]
        logger.info(f"[Chunk] total_pairs={n_pairs}, tokens_per_chunk={self.tokens_per_chunk}")

    # ---------- 公共 API ----------
    def load_video(self, src_video_path: str):
        """
        上传视频后调用：一次性完成
        - ffmpeg 转码到 5FPS + 限定分辨率
        - 载入 TCHW 到 GPU
        - 提取/载入 16kHz WAV
        - 预计算全量 video/audio 连续嵌入
        - 重置时间/指针
        """
        self.reset()
        cache_path = self._video_preprocess(src_video_path)
        self._load_whole_video_tensor(cache_path)
        self._extract_and_load_audio(src_video_path)
        self._precompute_all_features()

        # 初始化对话：写入 system prompt 到 KV
        with torch.no_grad():
            sys_embeds = self.model.get_input_embeddings()(self._system_input_ids)
            out = self.model(inputs_embeds=sys_embeds, use_cache=True)
            self.past_key_values = out.past_key_values
            self.last_input_ids = self._system_input_ids[:, -1:]  # 记录最后一个 token
        self.time_ptr = 0.0
        self.v_ptr = 0
        self.a_ptr = 0
        logger.info("[Load] Ready for streaming.")

    def input_video_stream(self, wall_time: float):
        """
        前端定时喂当前 video time（秒）；当到达一个新 chunk 的结束边界 (~0.4s) 时，推一次。
        """
        # 计算现在应该处理到第几个 chunk
        target_chunk = int((wall_time + 1e-6) // self.cfg.seconds_per_chunk)
        while self.v_ptr < len(self.v_chunks) and self.a_ptr < len(self.a_chunks) and \
              self.v_ptr < target_chunk and self.a_ptr < target_chunk:
            self._consume_one_chunk()
        self.time_ptr = wall_time

    def _consume_one_chunk(self):
        """
        取出当前 chunk 的 (v_s:e, a_s:e) token 嵌入，拼接在上一次最后 token embed 后面，
        前向一步，检查下一 token 的分布：
          - 若 argmax==',' 且 p>=阈值 → 认为沉默，仅推进 last_input_ids，不触发生成
          - 否则 → 触发一次短的 greedy 续写，直到遇到换行/EOS/']'等自然停点（保守做法），
                  然后把生成文本解析/裁剪为一条 JSON（鲁棒正则），append 到 self.generated
        """
        (vs, ve) = self.v_chunks[self.v_ptr]
        (as_, ae) = self.a_chunks[self.a_ptr]
        v_emb = self.video_embeds_flat[vs:ve, :].to(self.device)
        a_emb = self.audio_embeds_flat[as_:ae, :].to(self.device)

        # 组装本次输入：上一次最后一个 token 的 embedding + [video chunk] + [audio chunk]
        last_emb = self.model.get_input_embeddings()(self.last_input_ids)
        step_embeds = torch.cat([last_emb, v_emb.unsqueeze(0), a_emb.unsqueeze(0)], dim=1).to(self.device)
        with torch.no_grad():
            out = self.model(inputs_embeds=step_embeds, use_cache=True, past_key_values=self.past_key_values)
        self.past_key_values = out.past_key_values

        # 看最后一个位置的 logit
        next_logits = out.logits[:, -1, :]  # [1, V]
        next_probs = torch.softmax(next_logits, dim=-1)
        p_comma = next_probs[0, self.cfg.comma_token_id].item()
        next_id = int(torch.argmax(next_probs, dim=-1)[0].item())

        # Debug
        logger.info(f"[Chunk #{self.v_ptr}] v[{vs}:{ve}] a[{as_}:{ae}] "
                    f"next_id={next_id} p_comma={p_comma:.3f}")

        if next_id == self.cfg.comma_token_id and p_comma >= self.cfg.comma_silence_threshold:
            # 沉默：推进状态，更新 last_input_ids（就是逗号）
            self.last_input_ids = torch.tensor([[self.cfg.comma_token_id]], device=self.device)
            self.v_ptr += 1
            self.a_ptr += 1
            return

        # 触发“说话”：把','的概率归零，再贪心解码一小段
        forced_logits = next_logits.clone()
        forced_logits[:, self.cfg.comma_token_id] = -1e9
        gen_id = int(torch.argmax(forced_logits, dim=-1)[0].item())
        self.last_input_ids = torch.tensor([[gen_id]], device=self.device)

        # 继续生成，直到遇到一个停点（安全限长）
        gen_ids = [gen_id]
        max_new = 128
        for _ in range(max_new - 1):
            emb = self.model.get_input_embeddings()(self.last_input_ids)
            with torch.no_grad():
                out2 = self.model(inputs_embeds=emb, use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = out2.past_key_values
            logits2 = out2.logits[:, -1, :]
            probs2 = torch.softmax(logits2, dim=-1)
            # 简单贪心
            tid = int(torch.argmax(probs2, dim=-1)[0].item())
            gen_ids.append(tid)
            self.last_input_ids = torch.tensor([[tid]], device=self.device)

            # 停点启发：遇到换行、eos 或 右中括号（很多JSON会以 ] 或 } 结束）
            if tid in (self.model.config.eos_token_id,):
                break

        text = self.processor.batch_decode(torch.tensor([gen_ids], device=self.device),
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)[0]
        text = text.strip()
        logger.info(f"[GEN] {text}")

        # 粗暴截取 JSON 片段（容错）
        json_piece = self._extract_json_object(text)
        if json_piece:
            self.generated.append(json_piece)

        self.v_ptr += 1
        self.a_ptr += 1

    @staticmethod
    def _extract_json_object(txt: str) -> Optional[str]:
        """
        尝试从 txt 中抽出一条 { ... } 或 [ { ... } ] 片段；失败就直接返回原文本。
        """
        txt = txt.strip()
        # 简单两段式尝试
        if txt.startswith("{"):
            r = txt.split("}", 1)
            if len(r) >= 2:
                return r[0] + "}"
        if txt.startswith("["):
            r = txt.split("]", 1)
            if len(r) >= 2:
                return r[0] + "]"
        # 尝试回落：如果本身就是短 JSON-like，就原样返回
        if ("emotion" in txt and "summary_reasoning" in txt):
            return txt
        return None

    # ---------- Gradio 接口配套 ----------
    def feed_and_maybe_respond(self, video_time: float) -> Optional[str]:
        """
        由前端（每 0.4s）调用：消耗到当前 time 所需的 chunk；如果本轮有新 JSON 片段，就返回增量。
        """
        prev_n = len(self.generated)
        self.input_video_stream(video_time)
        if len(self.generated) > prev_n:
            # 只返回本次新增
            return self.generated[-1]
        return None
