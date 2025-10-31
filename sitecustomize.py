# /home/CORP/zhuo.zhi/Project/Qwen2.5-Omni-EMO/sitecustomize.py
from transformers import AutoConfig, AutoModelForCausalLM

# 这些类名是你本地 transformers 里已经存在的自定义实现（你之前能在 demo 里加载成功，说明它们可用）
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)

# 注册 config 映射：'qwen2_5_omni_thinker' -> Qwen2_5OmniThinkerConfig
try:
    AutoConfig.register("qwen2_5_omni_thinker", Qwen2_5OmniThinkerConfig)
except Exception:
    pass

# 注册模型类映射：Config -> ForCausalLM
try:
    AutoModelForCausalLM.register(Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration)
except Exception:
    pass
