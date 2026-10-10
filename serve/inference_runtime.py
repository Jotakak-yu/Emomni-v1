"""Batch-one Qwen2 inference state, scoped to the current conversation."""
from collections import OrderedDict
import hashlib
import math
import os
from pathlib import Path

import torch
from transformers import CompileConfig, DynamicCache, StaticCache


class InferenceRuntime:
    def __init__(self, model, extractor, device):
        self.model, self.extractor, self.device = model, extractor, device
        self.audio_enabled = os.getenv('EMOMNI_AUDIO_CACHE', '1') == '1'
        self.prefix_enabled = os.getenv('EMOMNI_PREFIX_CACHE', '1') == '1'
        self.compile_enabled = os.getenv('EMOMNI_DECODE_COMPILE', '0') == '1'
        self.audio = OrderedDict()
        self.session = None
        self.cache = None
        self.keys = []
        self.capacity = 0

    def begin(self, session):
        # No implicit cross-conversation cache reuse, including API callers
        # which do not provide a stable session id.
        if session is None or session != self.session:
            self.clear()
            self.audio.clear()
        self.session = session

    @torch.inference_mode()
    def clear(self):
        self.keys = []
        if self.cache is not None and isinstance(self.cache, StaticCache):
            self.cache.reset()
        else:
            self.cache = None

    @torch.inference_mode()
    def speech(self, path):
        from .audio import load_audio
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if self.audio_enabled and digest in self.audio:
            self.audio.move_to_end(digest)
            return self.audio[digest]
        waveform = load_audio(path, self.extractor.sampling_rate)
        features = self.extractor(waveform, sampling_rate=self.extractor.sampling_rate,
                                  return_attention_mask=True, return_tensors='pt')
        embeds, mask, *_ = self.model.get_speech_features(
            features.input_features.to(self.device, dtype=torch.bfloat16),
            features.attention_mask.to(self.device))
        result = (embeds.detach(), mask.detach(), digest)
        if self.audio_enabled:
            self.audio[digest] = result
            while len(self.audio) > 16:
                self.audio.popitem(last=False)
        return result

    @torch.inference_mode()
    def chat(self, history, generation_config, streamer=None, stopping_criteria=None):
        embeddings, keys = [], []
        for segment in history:
            if len(segment) == 1:
                ids = segment[0].to(self.device)
                embeddings.append(self.model.qwen_model.get_input_embeddings()(ids))
                keys.extend(segment[0][0].tolist())
            elif len(segment) == 3:
                embeds, mask, digest = segment
                embeddings.append(embeds)
                keys.extend((digest, i) for i in range(embeds.shape[1]))
            else:
                raise ValueError('Expected token ids or cached speech embeddings')
        inputs = torch.cat(embeddings, dim=1)
        length = inputs.shape[1]
        reuse = 0
        if self.prefix_enabled:
            for old, new in zip(self.keys, keys[:-1]):
                if old != new:
                    break
                reuse += 1
        required = length + generation_config.max_new_tokens
        if self.compile_enabled:
            capacity = 256 * math.ceil(required / 256)
            if not isinstance(self.cache, StaticCache) or self.capacity < required:
                self.cache = StaticCache(config=self.model.qwen_model.config, max_cache_len=capacity)
                self.capacity = capacity
                reuse = 0
            else:
                for layer in self.cache.layers:
                    if layer.is_initialized:
                        layer.keys[:, :, reuse:].zero_()
                        layer.values[:, :, reuse:].zero_()
            generation_config.compile_config = CompileConfig(mode='reduce-overhead', fullgraph=True)
            generation_config.disable_compile = False
        else:
            if self.cache is None or not isinstance(self.cache, DynamicCache):
                self.cache = DynamicCache(config=self.model.qwen_model.config)
                reuse = 0
            else:
                self.cache.crop(reuse)
            generation_config.disable_compile = True
        generation_config.return_dict_in_generate = True
        try:
            result = self.model.qwen_model.generate(
                inputs_embeds=inputs,
                attention_mask=torch.ones((1, length), dtype=torch.long, device=self.device),
                cache_position=torch.arange(reuse, length, device=self.device),
                past_key_values=self.cache,
                generation_config=generation_config, streamer=streamer,
                stopping_criteria=stopping_criteria)
            self.cache = result.past_key_values
            # The final generated token has not yet been forwarded through Qwen.
            generated = result.sequences[0].tolist()
            self.keys = (keys + generated)[:length + max(0, len(generated) - 1)]
            return result.sequences, length
        except BaseException:
            self.clear()
            raise
