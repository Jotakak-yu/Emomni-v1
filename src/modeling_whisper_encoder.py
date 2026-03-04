from typing import Optional, Tuple
from dataclasses import dataclass
import os

import torch
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder as HFWhisperEncoder
from transformers.utils import ModelOutput
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file

@dataclass
class WhisperOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    output_lengths: Optional[torch.LongTensor] = None


class WhisperEncoder(HFWhisperEncoder):
    """
    overwrite forward to support attention_mask
    overwrite from_pretrained to support split encoder parameters from pretrained WhisperModel
    """

    @staticmethod
    def from_pretrained(model_path, torch_dtype=None):
        config = WhisperConfig.from_pretrained(model_path)
        # Ensure a valid attention implementation for Whisper encoder
        if getattr(config, "_attn_implementation", None) is None:
            config.attn_implementation = "eager"

        model = WhisperEncoder(config)

        # Try loading from safetensors first (single-file or sharded), then fallback to .bin
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        bin_index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        safetensors_path = os.path.join(model_path, "model.safetensors")
        safetensors_index_path = os.path.join(model_path, "model.safetensors.index.json")

        state_dict = {}

        if os.path.isfile(safetensors_path):
            # Single safetensors file
            tensor_dict = safe_load_file(safetensors_path)
            for name, tensor in tensor_dict.items():
                if "model.encoder." in name:
                    new_name = name.replace("model.encoder.", "")
                    state_dict[new_name] = tensor
        elif os.path.isfile(safetensors_index_path):
            # Sharded safetensors via index: map keys to shard files then open shards
            import json
            with open(safetensors_index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            shard_to_keys = {}
            for key, shard_file in weight_map.items():
                if "model.encoder." in key:
                    shard_to_keys.setdefault(shard_file, []).append(key)
            for shard_file, keys in shard_to_keys.items():
                shard_path = os.path.join(model_path, shard_file)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for k in keys:
                        if k in f.keys():
                            tensor = f.get_tensor(k)
                            new_name = k.replace("model.encoder.", "")
                            state_dict[new_name] = tensor
        elif os.path.isfile(bin_path):
            # Legacy .bin single-file
            old_state_dict = torch.load(bin_path, map_location="cpu")
            for para_name in old_state_dict.keys():
                if "model.encoder." in para_name:
                    new_name = para_name.replace("model.encoder.", "")
                    state_dict[new_name] = old_state_dict[para_name]
        elif os.path.isfile(bin_index_path):
            # Sharded .bin (PyTorch). Load each shard referenced by index
            import json
            with open(bin_index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            shard_to_keys = {}
            for key, shard_file in weight_map.items():
                if "model.encoder." in key:
                    shard_to_keys.setdefault(shard_file, []).append(key)
            for shard_file, keys in shard_to_keys.items():
                shard_path = os.path.join(model_path, shard_file)
                shard_sd = torch.load(shard_path, map_location="cpu")
                for k in keys:
                    if k in shard_sd:
                        new_name = k.replace("model.encoder.", "")
                        state_dict[new_name] = shard_sd[k]
        else:
            raise FileNotFoundError(
                "No model weights found. Expected one of: model.safetensors, model.safetensors.index.json, pytorch_model.bin, pytorch_model.bin.index.json"
            )

        model.load_state_dict(state_dict)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return model

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = super().forward(
            input_features,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        last_hidden_state = output.last_hidden_state # B x T x C
        input_lengths = attention_mask.sum(-1)
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
        max_length = output_lengths.max()
        last_hidden_state = last_hidden_state[:,:max_length,:]

        return WhisperOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None,
            output_lengths=output_lengths
        )
