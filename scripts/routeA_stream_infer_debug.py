#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline inference script for emotion recognition model.
Processes video+audio in chunks and detects emotion changes.
"""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def extract_audio_from_video(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video using ffmpeg."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-sample_fmt", "s16", str(audio_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✓ Extracted audio to: {audio_path}")


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def load_and_process_media(
    video_path: str, 
    audio_path: str, 
    processor
) -> Tuple[Dict, Dict, List[int]]:
    """Load and process video and audio."""
    import av
    import librosa
    
    print("Loading video...")
    container = av.open(video_path, "r")
    video_stream = next(s for s in container.streams if s.type == "video")
    
    video_fps = 5.0
    video_maxlen = 300
    total_frames = video_stream.frames
    
    if total_frames > 0:
        duration = float(video_stream.duration * video_stream.time_base)
        sample_frames = max(1, math.floor(duration * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
    else:
        sample_indices = np.arange(video_maxlen)
    
    container.seek(0)
    frames = []
    for frame_idx, frame in enumerate(container.decode(video_stream)):
        if frame_idx in sample_indices:
            img = frame.to_image()
            if img.mode != "RGB":
                img = img.convert("RGB")
            frames.append(img)
    container.close()
    
    if len(frames) % 2 != 0:
        frames.append(frames[-1])
    
    print(f"✓ Loaded {len(frames)} video frames")
    
    print("Loading audio...")
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"✓ Loaded audio: {len(audio)/sr:.2f}s")
    
    # Process video
    video_inputs = processor.video_processor(
        images=None, videos=[frames],
        min_pixels=36864, max_pixels=36864
    )
    
    temporal_patch_size = processor.video_processor.temporal_patch_size
    video_second_per_grid = [temporal_patch_size / video_fps]
    video_inputs["video_second_per_grid"] = torch.tensor(video_second_per_grid)
    
    # Process audio
    audio_inputs = processor.feature_extractor(
        audio, sampling_rate=16000,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )
    audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
    input_lengths = (audio_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
    audio_lengths = [(input_lengths[0] - 2) // 2 + 1]
    
    return video_inputs, audio_inputs, audio_lengths


def get_chunk_tokens(
    video_inputs: Dict,
    audio_inputs: Dict,
    audio_lengths: List[int],
    processor,
    chunk_idx: int
) -> str:
    """Get multimodal tokens for a specific chunk."""
    T, H, W = video_inputs["video_grid_thw"][0].tolist()
    video_spg = video_inputs["video_second_per_grid"][0].item()
    audio_length = audio_lengths[0]
    
    position_id_per_seconds = 25
    seconds_per_chunk = 0.4
    t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
    merge_size = processor.video_processor.merge_size
    v_tokens_per_grid = (H // merge_size) * (W // merge_size)
    
    # Build indices
    video_t_index = torch.arange(T) * video_spg * position_id_per_seconds
    video_t_index = video_t_index.long().view(-1, 1).expand(-1, v_tokens_per_grid).reshape(-1)
    audio_t_index = torch.arange(audio_length).long()
    
    # Get chunks
    video_chunks = get_chunked_index(video_t_index, t_ntoken_per_chunk)
    audio_chunks = get_chunked_index(audio_t_index, t_ntoken_per_chunk)
    
    if chunk_idx >= len(video_chunks) or chunk_idx >= len(audio_chunks):
        return None
    
    # Get this chunk's tokens
    sv, ev = video_chunks[chunk_idx]
    nv = ev - sv
    sa, ea = audio_chunks[chunk_idx]
    na = ea - sa
    
    tokens = []
    if nv > 0:
        tokens.extend([
            processor.tokenizer.vision_bos_token,
            processor.tokenizer.video_token * int(nv),
            processor.tokenizer.vision_eos_token
        ])
    if na > 0:
        tokens.extend([
            processor.tokenizer.audio_bos_token,
            processor.tokenizer.audio_token * int(na),
            processor.tokenizer.audio_eos_token
        ])
    
    return "".join(tokens)


def get_chunked_index(token_indices: torch.Tensor, tokens_per_chunk: int) -> List[Tuple[int, int]]:
    """Split token indices into chunks."""
    token_indices = token_indices.numpy()
    chunks = []
    i, start_idx = 0, 0
    current_chunk = 1
    
    while i < len(token_indices):
        if token_indices[i] >= current_chunk * tokens_per_chunk:
            if i > start_idx:
                chunks.append((int(start_idx), int(i)))
            start_idx = i
            current_chunk += 1
        i += 1
    
    if start_idx < len(token_indices):
        chunks.append((int(start_idx), int(len(token_indices))))
    
    return chunks


def parse_emotion_json(text: str) -> Dict:
    """Parse emotion JSON from generated text."""
    # Try to find JSON pattern
    json_match = re.search(r'\{[^}]*"emotion"[^}]*\}', text)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            if "emotion" in obj:
                return obj
        except:
            pass
    return None


def streaming_inference(
    model,
    processor,
    video_inputs: Dict,
    audio_inputs: Dict,
    audio_lengths: List[int],
    user_prompt: str,
    device: str,
    max_chunks: int = None
) -> List[Dict]:
    """Run streaming inference chunk by chunk."""
    print(f"\n{'='*80}")
    print("STARTING INFERENCE")
    print(f"{'='*80}\n")
    
    # Move to device
    for k in ["pixel_values_videos", "video_grid_thw", "video_second_per_grid"]:
        if k in video_inputs and isinstance(video_inputs[k], torch.Tensor):
            video_inputs[k] = video_inputs[k].to(device)
    for k in ["input_features", "feature_attention_mask"]:
        if k in audio_inputs and isinstance(audio_inputs[k], torch.Tensor):
            audio_inputs[k] = audio_inputs[k].to(device)
    
    # Build initial prompt
    messages = [{"role": "user", "content": user_prompt}]
    base_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if isinstance(base_text, list):
        base_text = base_text[0]
    
    current_text = base_text
    results = []
    
    # Get total chunks
    T = video_inputs["video_grid_thw"][0][0].item()
    total_chunks = T if max_chunks is None else min(T, max_chunks)
    
    print(f"Total chunks to process: {total_chunks}\n")
    
    comma_id = processor.tokenizer.encode(",", add_special_tokens=False)[0]
    
    for chunk_idx in range(total_chunks):
        # Get tokens for this chunk
        chunk_tokens = get_chunk_tokens(
            video_inputs, audio_inputs, audio_lengths,
            processor, chunk_idx
        )
        if not chunk_tokens:
            break
        
        # Add comma before chunk (except first)
        if chunk_idx > 0:
            current_text += ","
        
        current_text += chunk_tokens
        
        # Tokenize
        text_inputs = processor.tokenizer(current_text, return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        
        # Get position IDs
        get_rope = model.get_rope_index if hasattr(model, "get_rope_index") else model.model.get_rope_index
        position_ids, rope_deltas = get_rope(
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=video_inputs["video_grid_thw"],
            attention_mask=(attention_mask >= 1).float(),
            second_per_grids=video_inputs["video_second_per_grid"]
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rope_deltas=rope_deltas - (1 - (attention_mask >= 1).float()).sum(dim=-1).unsqueeze(-1),
                pixel_values_videos=video_inputs["pixel_values_videos"],
                video_grid_thw=video_inputs["video_grid_thw"],
                input_features=audio_inputs["input_features"],
                feature_attention_mask=audio_inputs["feature_attention_mask"],
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode generated part
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse result
        if len(generated_ids) == 0:
            print(f"Chunk {chunk_idx+1}/{total_chunks}: (no generation)")
            break
        
        first_token_id = generated_ids[0].item()
        
        # Check if it's a comma or JSON
        emotion_obj = parse_emotion_json(generated_text)
        
        if emotion_obj:
            # Emotion change detected
            emotion = emotion_obj.get("emotion", "unknown")
            reasoning = emotion_obj.get("summary_reasoning", "")
            print(f"Chunk {chunk_idx+1:3d}/{total_chunks}: ✓ {emotion:12s} - {reasoning[:60]}...")
            results.append(emotion_obj)
            current_text += generated_text.split("}")[0] + "}"  # Add clean JSON
        else:
            # No change, just comma
            print(f"Chunk {chunk_idx+1:3d}/{total_chunks}: , (no change)")
            current_text += ","
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Offline emotion recognition inference")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--prompt_path", required=True, help="Path to user prompt file")
    parser.add_argument("--output_json", default=None, help="Path to save results JSON")
    parser.add_argument("--max_chunks", type=int, default=None, help="Max chunks to process")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--temp_audio", default="/tmp/offline_inference_audio.wav", help="Temp audio path")
    args = parser.parse_args()
    
    # Load prompt
    with open(args.prompt_path) as f:
        user_prompt = f.read().strip()
    
    # Extract audio
    audio_path = Path(args.temp_audio)
    print(f"Extracting audio from video...")
    extract_audio_from_video(Path(args.video), audio_path)
    
    # Get video info
    duration = get_video_duration(Path(args.video))
    print(f"Video duration: {duration:.2f}s\n")
    
    # Load model and processor
    print(f"Loading model from: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded\n")
    
    # Load and process media
    video_inputs, audio_inputs, audio_lengths = load_and_process_media(
        args.video, str(audio_path), processor
    )
    
    # Run inference
    results = streaming_inference(
        model, processor,
        video_inputs, audio_inputs, audio_lengths,
        user_prompt, args.device,
        max_chunks=args.max_chunks
    )
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Total emotion changes detected: {len(results)}\n")
    
    if results:
        for i, result in enumerate(results, 1):
            emotion = result.get("emotion", "unknown")
            reasoning = result.get("summary_reasoning", "")
            print(f"[{i}] {emotion}: {reasoning}")
    else:
        print("No emotion changes detected.")
    
    # Save to file if specified
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    # Cleanup
    if audio_path.exists():
        audio_path.unlink()
    
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
'''
CUDA_VISIBLE_DEVICES=0 \
python /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/scripts/routeA_stream_infer_debug.py \
  --model /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora/merged \
  --video /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/LLaMA-Factory/data/dataset/lemon/video/vid_0111_clip16.mp4 \
  --prompt_path /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/user_prompt.txt \
  --step 0.4
'''

'''
CUDA_VISIBLE_DEVICES=0 \
python /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/scripts/routeA_stream_infer_debug.py \
    --model /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora_1104/merged \
    --video /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/LLaMA-Factory/data/dataset/lemon/video/vid_0111_clip16.mp4 \
    --prompt_path /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/user_prompt.txt \
    --max_chunks 100
'''