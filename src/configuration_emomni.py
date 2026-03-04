"""Emomni config"""

from transformers import PretrainedConfig, WhisperConfig, AutoConfig
from transformers import logging
from peft import LoraConfig

logger = logging.get_logger(__name__)

class EmomniConfig(PretrainedConfig):
    model_type = "emomni-v1"
    has_no_defaults_at_init = True

    def __init__(
        self, 
        whisper_config=None, 
        qwen_config=None,
        conv_kernel_sizes="5,5,5",
        adapter_inner_dim=512,
        adapter_hidden_layers=0,
        kd_temperature=2,
        kd_smoothing_weight=0.5,
        tie_embedding=None,
        lora_config={},
        lora_scope="audio",
        adapter_type="subsampler",  # choose from "subsampler" or "cformer"
        num_pre_cif_layers=4,
        num_post_cif_layers=4,
        num_emotions=5,
        **kwargs
    ):
        super().__init__(**kwargs)

        if whisper_config is None:
            whisper_config = {}
            logger.info("whisper config is None. Initializing the WhisperConfig with default values")

        if qwen_config is None:
            qwen_config = {}
            logger.info("qwen config is None. Initializing the QwenConfig with default values")

        self.whisper_config = WhisperConfig(**whisper_config).to_dict()
        
        # 处理qwen_config，确保能正确获取_name_or_path
        resolved_name_or_path = None
        if isinstance(qwen_config, dict):
            # 优先从字典中取值
            resolved_name_or_path = qwen_config.get('_name_or_path')
            if resolved_name_or_path is None:
                # 回退：从kwargs中获取训练/推理入口传入的qwen_model
                resolved_name_or_path = kwargs.get('qwen_model', None) or kwargs.get('_name_or_path', None)
        else:
            # 兼容直接传入AutoConfig的情况
            resolved_name_or_path = getattr(qwen_config, '_name_or_path', None)

        if not resolved_name_or_path:
            raise ValueError(
                "qwen_config 缺少基础LLM路径。请通过训练入口 --qwen_model 传入，或在 qwen_config 中包含 '_name_or_path' 字段。"
            )

        # 规范化为AutoConfig对象并记录路径
        if isinstance(qwen_config, dict):
            qwen_config_clean = {k: v for k, v in qwen_config.items() if k != '_name_or_path'}
            self.qwen_config = AutoConfig.from_pretrained(resolved_name_or_path, **qwen_config_clean)
        else:
            self.qwen_config = qwen_config
        self._name_or_path = resolved_name_or_path
            
        self.lora_config = lora_config
        self.lora_scope = lora_scope

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim
        self.adapter_hidden_layers = adapter_hidden_layers
        self.kd_temperature = kd_temperature
        # Default tie_embedding to Qwen's tie_word_embeddings when not explicitly provided
        if tie_embedding is None:
            self.tie_embedding = bool(getattr(self.qwen_config, "tie_word_embeddings", False))
        else:
            self.tie_embedding = bool(tie_embedding)

        self.adapter_type = adapter_type
        self.num_pre_cif_layers = num_pre_cif_layers
        self.num_post_cif_layers = num_post_cif_layers
        self.num_emotions = num_emotions