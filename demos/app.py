# -*- coding: utf-8 -*-
import time
import traceback
import gradio as gr
import transformers
from .live_infer import LiveInfer, InferConfig

transformers.logging.set_verbosity_info()

CSS = """
#gr_title {text-align:center;}
#gr_video {max-height: 540px;}
#gr_jsonbox {max-height: 540px;}
"""

def _append_log(old: str, line: str) -> str:
    ts = time.strftime("%H:%M:%S")
    new = (old or "") + f"[{ts}] {line}\n"
    return new[-4000:]  # 只保留最后一些，避免无限增长

def build_demo(model_path: str):
    infer = LiveInfer(InferConfig(model_path=model_path))
    SEC_PER_CHUNK = float(getattr(infer.cfg, "seconds_per_chunk", 0.4))  # 你的训练就是 0.4

    with gr.Blocks(title="EMO-Stream Demo (Qwen2.5-Omni)", css=CSS) as demo:
        gr.Markdown("# Qwen2.5-Omni Streaming Emotion Demo", elem_id="gr_title")

        with gr.Row():
            with gr.Column():
                gr_video = gr.Video(
                    label="Upload a video (with audio). Autoplay on.",
                    elem_id="gr_video",
                    visible=True,
                    sources=["upload"],
                    autoplay=True,
                )
                gr.Markdown(
                    "- We process the whole video once to build streaming chunks (0.4s).\n"
                    "- The model will **stay silent** (comma) most of the time, and **speak** only when an emotion change is detected."
                )
                dbg = gr.Textbox(label="Debug log (append only)", interactive=False)

            with gr.Column():
                jsonbox = gr.Chatbot(
                    label="Streaming JSON output",
                    elem_id="gr_jsonbox",
                    type="messages",  # 消除 deprecation 警告
                    render=True,
                )

                threshold = gr.Slider(
                    minimum=0.5,
                    maximum=0.95,
                    step=0.01,
                    value=infer.cfg.comma_silence_threshold,
                    label="Silence threshold (p(','))",
                )

                def on_thr(x):
                    infer.cfg.comma_silence_threshold = float(x)
                    return f"Silence threshold set to {x:.2f}"

                thr_msg = gr.Markdown("")
                threshold.change(on_thr, inputs=threshold, outputs=thr_msg)

        # 状态：gate控制生成器循环；ticks是已消耗chunk数；vtime是“虚拟时间”(秒)
        gate = gr.State(False)
        tick_count = gr.State(0)
        vtime = gr.State(0.0)

        # 1) 上传/切换视频
        def on_video_change(vpath, history, _gate, _ticks, _vtime, _dbg):
            if not vpath:
                return history, False, 0, 0.0, _append_log(_dbg, "[WARN] No video.")
            try:
                _dbg = _append_log(_dbg, "[INFO] Loading video...")
                infer.load_video(vpath)
                _dbg = _append_log(_dbg, "[OK] video loaded & features precomputed.")
            except Exception as e:
                tb = traceback.format_exc(limit=1)
                return (
                    history,
                    False,
                    0,
                    0.0,
                    _append_log(_dbg, f"[ERROR] load_video failed: {e}\n{tb}"),
                )

            # 初始化右侧历史
            history = [{"role": "system", "content": "ready"}]
            # 重置计数与虚拟时间
            return history, True, 0, 0.0, _dbg

        gr_video.change(
            on_video_change,
            inputs=[gr_video, jsonbox, gate, tick_count, vtime, dbg],
            outputs=[jsonbox, gate, tick_count, vtime, dbg],
        )

        # 2) 后端生成器：每 ~0.4s 推进一次“虚拟时间”，驱动 chunk 消费
        def refresher(history, _gate, _ticks, _vtime, _dbg):
            if not _gate:
                yield history, _gate, _ticks, _vtime, _append_log(_dbg, "[INFO] refresher: gate is False, idle.")
                return

            _dbg = _append_log(_dbg, "[INFO] refresher: started.")
            yield history, _gate, _ticks, _vtime, _dbg

            while _gate:
                try:
                    # 计算本次要发送的“虚拟时间”
                    t_sec = float(_vtime or 0.0)
                    # 调用推理：按 0.4s/chunk 消耗
                    incr = infer.feed_and_maybe_respond(t_sec)

                    _ticks = int((_ticks or 0)) + 1
                    _vtime = t_sec + SEC_PER_CHUNK

                    if incr:
                        history = history + [{"role": "assistant", "content": incr}]
                        _dbg = _append_log(_dbg, f"[Output @ {t_sec:.2f}s] {incr}")
                    else:
                        _dbg = _append_log(_dbg, f"[Tick {_ticks} @ {t_sec:.2f}s] (no output)")

                    # 推送到前端
                    yield history, _gate, _ticks, _vtime, _dbg

                    # 睡一小会儿，基本与 0.4s 同步；留余量避免阻塞UI
                    time.sleep(max(SEC_PER_CHUNK - 0.02, 0.05))

                    # 如果有结束标记就停（live_infer.py 若实现了 is_finished）
                    if hasattr(infer, "is_finished") and infer.is_finished():
                        _dbg = _append_log(_dbg, "[INFO] Reached end-of-stream. Stop refresher.")
                        _gate = False
                        yield history, _gate, _ticks, _vtime, _dbg
                        break

                except Exception as e:
                    tb = traceback.format_exc(limit=1)
                    history = history + [{"role": "assistant", "content": f'{{"error": "{e}"}}'}]
                    _dbg = _append_log(_dbg, f"[ERROR] refresher exception: {e}\n{tb}")
                    yield history, False, _ticks, _vtime, _dbg
                    break

        # gate=True 时启动生成器循环
        gate.change(
            refresher,
            inputs=[jsonbox, gate, tick_count, vtime, dbg],
            outputs=[jsonbox, gate, tick_count, vtime, dbg],
        )

        demo.queue()
    return demo


if __name__ == "__main__":
    MODEL_PATH = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora/merged"
    demo = build_demo(MODEL_PATH)
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
