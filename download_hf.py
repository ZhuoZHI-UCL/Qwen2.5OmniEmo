# from huggingface_hub import snapshot_download

# # 下载 PhilipC/HumanOmniV2 模型
# snapshot_download(
#     repo_id="Qwen/Qwen2.5-Omni-7B",        # 模型仓库名
#     repo_type="model",                    # ⚠️ 注意这里改为 "model"
#     local_dir="/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/hf_cache/Qwen2.5-Omni-7B",  # 本地保存目录
#     local_dir_use_symlinks=False
# )


import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import requests
from huggingface_hub import hf_hub_download, constants

# 全局关闭 SSL 验证
requests.adapters.DEFAULT_RETRIES = 3
constants._HF_HUB_HTTP_CLIENT = requests.Session()
constants._HF_HUB_HTTP_CLIENT.verify = False

import shutil

target_dir = "/home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/LLaMA-Factory/data/dataset/lemon/video"
cache_path = hf_hub_download(
    repo_id="UCEEZZZ/server",
    filename="data_clips.tar",
    repo_type="dataset"
)

shutil.copy(cache_path, f"{target_dir}/data_clips.tar")
print("✅ 已下载到目标路径")

