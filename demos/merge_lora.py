# merge_lora_offline.py
import os
from transformers import AutoProcessor
from transformers import Qwen2_5OmniThinkerForConditionalGeneration  # 关键：显式类，避免 Auto 走网
from peft import PeftModel
import torch
import json
import shutil

# ====== 配置你自己的路径 ======
BASE = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/hf_cache/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
LORA_PATH = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora/checkpoint-201"
MERGED_DIR = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/output/lemon_omni_lora/merged"

os.environ["HF_HUB_OFFLINE"] = "1"   # 双保险：彻底禁止联网

print("[1/4] Load base model (local only)...")
base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    BASE,
    trust_remote_code=True,
    local_files_only=True,   # 关键：只读本地缓存/路径
    torch_dtype="auto",
    device_map="cpu",
)

print("[2/4] Attach LoRA adapter (local only)...")
model = PeftModel.from_pretrained(
    base,
    LORA_PATH,
    local_files_only=True,
    is_trainable=False,
)

print("[3/4] Merge & unload...")
model = model.merge_and_unload()  # 合并成完整模型，去掉 LoRA 分支

os.makedirs(MERGED_DIR, exist_ok=True)
model.save_pretrained(MERGED_DIR)
print(f"  ✓ Model saved to: {MERGED_DIR}")

print("[4/4] Save processor/tokenizer...")
# 最稳妥：用“基座”的处理器（训练时你也是基座处理器+LoRA）
processor = AutoProcessor.from_pretrained(
    BASE,
    trust_remote_code=True,
    local_files_only=True,
)
processor.save_pretrained(MERGED_DIR)

# 如果 LoRA 目录里有新增的 special tokens（added_tokens.json 等），也拷过去（可选）
for fname in ["added_tokens.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
    src = os.path.join(LORA_PATH, fname)
    dst = os.path.join(MERGED_DIR, fname)
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {fname} from LoRA to merged (override if existed)")
        except Exception as e:
            print(f"  ! Skip copy {fname}: {e}")

print("Done.")
