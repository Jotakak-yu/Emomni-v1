import math
from typing import List, Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import logging
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperConfig

from .plora import LoraConfig, LoraModel
from .modeling_adapter import Subsampler, CFormer
from .configuration_emomni import EmomniConfig
from .modeling_utils import length_to_attention_mask, check_shape
from .modeling_whisper_encoder import WhisperEncoder
from transformers import AutoConfig
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

text_llm_related_losses = {"response_kl", "input_kl"}
speech_llm_related_losses = {"response_kl", "input_kl", "response_ce", "input_er"}
lm_related_losses = text_llm_related_losses | speech_llm_related_losses

class EmomniModel(PreTrainedModel):
    config_class = EmomniConfig
    base_model_prefix = "qwen_model"

    def __init__(self, config: EmomniConfig):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        # Ensure a valid attention implementation to avoid KeyError when building Whisper layers
        if getattr(self.whisper_config, "_attn_implementation", None) is None:
            self.whisper_config.attn_implementation = "eager"
        self.whisper_model = WhisperEncoder(self.whisper_config)

        # Enhanced Qwen model initialization with better 2.5/3 support
        self._init_qwen_model(config)
        
    def _init_qwen_model(self, config):
        """Enhanced Qwen model initialization with better Qwen2.5/3 support."""
        if isinstance(config.qwen_config, dict):
            # Prefer explicit _name_or_path in qwen_config, fallback to config._name_or_path set in EmomniConfig
            qwen_model_path = config.qwen_config.get('_name_or_path', '') or getattr(config, "_name_or_path", '')
            if not qwen_model_path:
                raise ValueError(
                    "qwen_model_path is empty. Ensure qwen_config._name_or_path or config._name_or_path is set. "
                    "For Qwen2.5/3 models, provide the base model path."
                )
            
            # Enhanced config loading with trust_remote_code for Qwen2.5/3
            try:
                qwen_config_dict = {k: v for k, v in config.qwen_config.items() if k != '_name_or_path'}
                self.qwen_config = AutoConfig.from_pretrained(
                    qwen_model_path, 
                    trust_remote_code=True,
                    **qwen_config_dict
                )
                logger.info(f"Loaded Qwen config from: {qwen_model_path}")
            except Exception as e:
                logger.error(f"Failed to load Qwen config from {qwen_model_path}: {e}")
                raise
        else:
            self.qwen_config = config.qwen_config
        
        # Enhanced model creation with better error handling
        try:
            self.qwen_model = AutoModelForCausalLM.from_config(
                self.qwen_config,
                trust_remote_code=True
            )
            logger.info(f"Created Qwen model: {self.qwen_config.model_type if hasattr(self.qwen_config, 'model_type') else 'unknown'}")
        except Exception as e:
            logger.error(f"Failed to create Qwen model: {e}")
            raise
        # Ensure lm_head is tied to input embeddings when enabled by either flag
        should_tie = getattr(config, "tie_embedding", False) or getattr(self.qwen_config, "tie_word_embeddings", False)
        if should_tie:
            self._ensure_tie_weights()

        if config.lora_config:
            self.lora_config = LoraConfig(**config.lora_config)
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")
            # Ensure weights remain tied after LoRA wrapping when enabled
            should_tie = getattr(config, "tie_embedding", False) or getattr(self.qwen_config, "tie_word_embeddings", False)
            if should_tie:
                self._ensure_tie_weights()

        if config.adapter_type == "subsampler":
            self.adapter = Subsampler(self.whisper_config.d_model, config.adapter_inner_dim, self.qwen_config.hidden_size,
                                      config.adapter_hidden_layers, self.whisper_config, config.conv_kernel_sizes)

        elif config.adapter_type == "cformer":
            self.adapter = CFormer(self.whisper_config, self.qwen_config.hidden_size,
                                   self.qwen_config.vocab_size,
                                   num_pre_cif_layers=config.num_pre_cif_layers,
                                   num_post_cif_layers=config.num_post_cif_layers)
        else:
            raise ValueError(f"unsupported adapter type: {config.adapter_type}")
        
        self.hidden2emotion = nn.Linear(self.qwen_config.hidden_size, self.config.num_emotions, bias=False)

        self.loss_names = [] # must be a list of loss names:  seq_kd, token_kd, or others before training

    def _ensure_tie_weights(self):
        """Enhanced weight tying with better Qwen2.5/3 support."""
        try:
            qwen = self.qwen_model
            
            # Check if we're dealing with a LoRA-wrapped model
            is_lora_wrapped = hasattr(qwen, 'base_model') or 'LoraModel' in str(type(qwen))
            if is_lora_wrapped:
                logger.debug("Detected LoRA-wrapped model, accessing base model for weight tying")
                base_model = getattr(qwen, 'base_model', qwen)
                if hasattr(base_model, 'model'):
                    qwen = base_model.model
                elif hasattr(base_model, 'base_model'):
                    qwen = base_model.base_model
            
            # Enhanced native tie_weights support
            if hasattr(qwen, "tie_weights"):
                try:
                    if hasattr(qwen, "config") and hasattr(qwen.config, "tie_word_embeddings"):
                        qwen.config.tie_word_embeddings = True
                    qwen.tie_weights()
                    logger.info("Successfully tied word embeddings via model.tie_weights()")
                    return
                except Exception as e:
                    logger.debug(f"Native tie_weights failed: {e}, trying manual approach")
            
            # Enhanced manual tying with better compatibility
            self._manual_tie_weights(qwen)
            
        except Exception as e:
            logger.warning(f"Failed to tie word embeddings: {e}")
    
    def _manual_tie_weights(self, qwen):
        """Manual weight tying with enhanced Qwen2.5/3 compatibility."""
        get_input = getattr(qwen, "get_input_embeddings", None)
        lm_head = getattr(qwen, "lm_head", None)
        
        # Try different attribute names for different Qwen versions
        if lm_head is None:
            for attr_name in ["lm_head", "output_projection", "classifier"]:
                if hasattr(qwen, attr_name):
                    lm_head = getattr(qwen, attr_name)
                    logger.debug(f"Found output layer: {attr_name}")
                    break
        
        if callable(get_input) and lm_head is not None:
            try:
                input_emb = get_input()
                if hasattr(input_emb, "weight") and hasattr(lm_head, "weight"):
                    # Enhanced shape compatibility check
                    input_shape = tuple(input_emb.weight.shape)
                    output_shape = tuple(lm_head.weight.shape)
                    
                    if input_shape != output_shape:
                        logger.warning(
                            f"Skip manual tie: shape mismatch input_emb={input_shape} vs lm_head={output_shape}"
                        )
                        return
                    
                    # Set config flag if available
                    if hasattr(qwen, "config") and hasattr(qwen.config, "tie_word_embeddings"):
                        qwen.config.tie_word_embeddings = True
                    
                    # Perform the tying
                    lm_head.weight = input_emb.weight
                    logger.info(f"Manually tied lm_head.weight to input embeddings (shapes: {input_shape})")
                else:
                    logger.debug("Could not access weight attributes for manual tying")
            except Exception as e:
                logger.warning(f"Manual weight tying failed: {e}")
        else:
            logger.debug("Could not find required components for manual weight tying")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Enhanced from_pretrained method with better Qwen2.5/3 support.

        Accepts an optional keyword 'qwen_model' used only to help resolve
        missing '_name_or_path' in saved configs. It is not a constructor arg
        and will be removed before delegating to HF.
        """
        # Extract helper-only arg if present
        qwen_model_helper = kwargs.pop("qwen_model", None)
        
        # Enhanced config loading with better error handling
        config = None
        try:
            # Load config first to check contents
            config = EmomniConfig.from_pretrained(
                pretrained_model_name_or_path,
            )
            logger.info(f"Loaded Emomni config from: {pretrained_model_name_or_path}")
            
            # Enhanced config resolution for Qwen2.5/3
            if isinstance(getattr(config, "qwen_config", None), dict):
                if not config.qwen_config.get("_name_or_path") and qwen_model_helper:
                    logger.info(f"Injecting Qwen model path: {qwen_model_helper}")
                    # Recreate EmomniConfig with injected _name_or_path so downstream init works
                    qwen_cfg = dict(config.qwen_config)
                    qwen_cfg["_name_or_path"] = qwen_model_helper
                    # Build a fresh config instance preserving other fields
                    config_kwargs = config.to_dict()
                    config_kwargs["qwen_config"] = qwen_cfg
                    # Remove attributes set by PretrainedConfig housekeeping
                    config_kwargs.pop("model_type", None)
                    config_kwargs.pop("transformers_version", None)
                    config = EmomniConfig(**config_kwargs)
                    
        except Exception as e:
            logger.warning(f"Failed to load config normally: {e}")
            if qwen_model_helper:
                logger.info(f"Attempting config creation with helper: {qwen_model_helper}")
            
        # Enhanced model loading with better error handling
        model = None
        try:
            if config:
                # Use prepared config
                model = super().from_pretrained(
                    pretrained_model_name_or_path, 
                    *model_args, 
                    config=config, 
                    **kwargs
                )
            else:
                # Fallback to default loading
                model = super().from_pretrained(
                    pretrained_model_name_or_path, 
                    *model_args, 
                    **kwargs
                )
            logger.info("Successfully loaded Emomni model")
            
        except Exception as e:
            logger.error(f"All loading attempts failed: {e}")
            raise e
        
        # Ensure tie_weights is applied after loading when enabled
        if model and (getattr(model.config, "tie_embedding", False) or getattr(model.qwen_config, "tie_word_embeddings", False)):
            try:
                model._ensure_tie_weights()
                logger.info("Applied weight tying")
            except Exception as e:
                logger.warning(f"Weight tying failed: {e}")
        
        return model

    def set_loss_names(self, names):
        self.loss_names = names

    def forward(
        self,
        start_ids: torch.LongTensor,
        start_mask: torch.Tensor,
        start_labels: torch.LongTensor,
        instruction_ids: torch.LongTensor,
        instruction_mask: torch.Tensor,
        instruction_labels: torch.LongTensor,
        audio_instruction_ids: torch.LongTensor,
        audio_instruction_mask: torch.Tensor,
        audio_instruction_labels: torch.LongTensor,
        input_ids: torch.LongTensor,
        input_mask: torch.Tensor,
        input_labels: torch.LongTensor,
        speech_values: torch.FloatTensor,
        speech_mask: torch.LongTensor,
        suffix_ids: torch.LongTensor,
        suffix_mask: torch.Tensor,
        suffix_labels: torch.LongTensor,
        emotion_labels: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        assert len(self.loss_names) > 0, "self.loss_names cannot be empty"

        if not any ("response" in loss_name for loss_name in self.loss_names):
            batch_size = start_ids.size(0)
            instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            audio_instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            audio_instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            audio_instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            suffix_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            suffix_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            suffix_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
        
        start_embeds = self.qwen_model.get_input_embeddings()(start_ids)
        instruction_embeds = self.qwen_model.get_input_embeddings()(instruction_ids)
        audio_instruction_embeds = self.qwen_model.get_input_embeddings()(audio_instruction_ids)
        input_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_ids)

        speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.get_speech_features(speech_values, speech_mask, input_mask.sum(-1))
        speech_input_labels = speech_input_mask.new_ones(speech_input_embeds.size(0), speech_input_embeds.size(1),
                                                         dtype=torch.int64).fill_(-100)

        speech_embeds = torch.cat([start_embeds, audio_instruction_embeds, speech_input_embeds, suffix_embeds], dim=1)
        speech_mask = torch.cat([start_mask, audio_instruction_mask, speech_input_mask, suffix_mask], dim=1)
        speech_labels = torch.cat([start_labels, audio_instruction_labels, speech_input_labels, suffix_labels], dim=1)

        if any(loss_name in text_llm_related_losses for loss_name in self.loss_names):
            text_embeds = torch.cat([start_embeds, instruction_embeds, input_embeds, suffix_embeds], dim=1)
            text_mask = torch.cat([start_mask, instruction_mask, input_mask, suffix_mask], dim=1)
            text_labels = torch.cat([start_labels, instruction_labels, input_labels, suffix_labels], dim=1)
            input_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                         torch.zeros_like(instruction_labels),
                                         input_mask,
                                         torch.zeros_like(suffix_labels)], dim=1)
            speech_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                          torch.zeros_like(audio_instruction_labels),
                                          speech_input_mask,
                                          torch.zeros_like(suffix_labels)], dim=1)
            text_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                 torch.zeros_like(instruction_labels),
                                                 torch.zeros_like(input_labels),
                                                 (suffix_labels != -100).long()], dim=1)
            speech_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                   torch.zeros_like(audio_instruction_labels),
                                                   torch.zeros_like(speech_input_labels),
                                                   (suffix_labels != -100).long()], dim=1)
            lora_audio_mask = torch.zeros_like(text_labels)
            self.update_lora_mask(lora_audio_mask, False)
            with torch.no_grad():
                text_output = self.qwen_model(inputs_embeds=text_embeds, attention_mask=text_mask,
                                              position_ids=text_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                              return_dict=True)
                text_logits = text_output.logits
        if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):
            lora_audio_mask = torch.cat([torch.zeros_like(start_mask),
                                         torch.zeros_like(audio_instruction_mask),
                                         torch.ones_like(speech_input_mask),
                                         torch.zeros_like(suffix_mask)], dim=1)
            self.update_lora_mask(lora_audio_mask, False)
            speech_output = self.qwen_model(inputs_embeds=speech_embeds, attention_mask=speech_mask,
                                            position_ids=speech_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                            return_dict=True)
            speech_logits = speech_output.logits

        total_loss = input_embeds.new_zeros(())
        for loss_name in self.loss_names:
            if loss_name == "response_ce":
                shifted_logits = speech_logits[..., :-1, :].contiguous()
                shifted_labels = speech_labels[..., 1:].contiguous()
                loss = F.cross_entropy(shifted_logits[shifted_labels != -100],
                                       shifted_labels[shifted_labels != -100], reduction="mean")
                total_loss += loss
            elif loss_name == "response_kl":
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_response_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[text_response_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    reduction="batchmean"
                )
                total_loss += loss
            elif loss_name == "input_kl":
                check_shape(input_labels, speech_input_labels)
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[input_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    reduction="batchmean"
                )
                total_loss += loss
            elif loss_name == "cif":
                if speech_pred_num_tokens is None:
                    raise RuntimeError("predicted_num_tokens not set but cif_loss is requested")
                loss = F.l1_loss(speech_pred_num_tokens/input_mask.sum(-1), torch.ones_like(speech_pred_num_tokens),
                                  reduction="mean")
                total_loss += loss
                # loss_str += f"{loss_name}: {loss.item():.4f}, "
            elif loss_name == "input_er":
                hidden_states = speech_input_embeds.clone()
                hidden_states[speech_input_mask == 0] = 0.0
                pooled_output = hidden_states.sum(dim=1) / speech_input_mask.sum(dim=1).view(-1, 1)
                er_logits = self.hidden2emotion(pooled_output)
                loss = F.cross_entropy(er_logits.view(-1, self.config.num_emotions), emotion_labels.view(-1))
                total_loss += loss
            else:
                raise RuntimeError(f"Unsupported loss name: {loss_name}")

        return {"loss": total_loss}

    def add_lora(self, lora_config, lora_scope="global"):
        """Enhanced LoRA addition with better Qwen2.5/3 support."""
        if self.config.lora_config:
            logger.warning(f"add_lora ignored as model already has lora enabled")
            return
        
        try:
            self.lora_config = lora_config
            self.config.lora_config = lora_config.to_dict()
            
            # Enhanced LoRA wrapping with better error handling
            original_model_type = type(self.qwen_model).__name__
            logger.info(f"Adding LoRA to {original_model_type} model")
            
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")
            self.config.lora_scope = lora_scope
            
            logger.info(f"Successfully added LoRA with scope: {lora_scope}")
            
            # Re-apply tie_weights after adding LoRA when enabled
            should_tie = getattr(self.config, "tie_embedding", False) or getattr(self.qwen_config, "tie_word_embeddings", False)
            if should_tie:
                try:
                    self._ensure_tie_weights()
                    logger.info("Re-applied weight tying after LoRA addition")
                except Exception as e:
                    logger.warning(f"Failed to re-tie weights after LoRA: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to add LoRA: {e}")
            raise

    def update_lora_mask(self, audio_mask, inference_mode: bool):
        if not self.config.lora_config or self.config.lora_scope == "global":
            return

        self.qwen_model.update_inference_mode(inference_mode)
        if self.config.lora_scope == "audio":
            self.qwen_model.update_lora_mask("default", audio_mask)
        elif self.config.lora_scope == "text":
            self.qwen_model.update_lora_mask("default", torch.ones_like(audio_mask) - audio_mask)
        elif self.config.lora_scope == "global":
            pass # do nonthing as official peft uses global lora
        else:
            raise ValueError(f"The scope value {self.config.lora_scope} for lora adapter 'default' is not supported")

    def merge_lora(self):
        """Enhanced LoRA merging with better error handling."""
        if not hasattr(self, 'lora_config'):
            raise ValueError("cannot call merge_lora when no self.lora_config is set")
        
        if self.config.lora_scope != "global":
            raise ValueError(
                f"cannot call merge_lora when the lora_scope is not global "
                f"(current scope: {self.config.lora_scope})"
            )
        
        try:
            logger.info("Merging LoRA adapters...")
            original_model_type = type(self.qwen_model).__name__
            
            self.qwen_model = self.qwen_model.merge_and_unload()
            self.config.lora_config = {}
            del self.lora_config
            
            logger.info(f"Successfully merged LoRA adapters back to {original_model_type}")
            
            # Ensure weights remain tied after merging adapters when enabled
            should_tie = getattr(self.config, "tie_embedding", False) or getattr(self.qwen_config, "tie_word_embeddings", False)
            if should_tie:
                try:
                    self._ensure_tie_weights()
                    logger.info("Re-applied weight tying after LoRA merge")
                except Exception as e:
                    logger.warning(f"Failed to re-tie weights after LoRA merge: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to merge LoRA: {e}")
            raise

    def get_speech_features(self, speech_values, speech_attention_mask, num_tokens=None):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        attention_mask = length_to_attention_mask(output.output_lengths)

        speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.adapter(speech_embeds, attention_mask, num_tokens)

        return speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        suffix_input_ids,
        suffix_attention_mask,
        speech_values=None,
        speech_attention_mask=None,
        generation_config=None
    ):
        inputs_embeds, input_attention_mask, lora_audio_mask = [], [], []

        prefix_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        inputs_embeds.append(prefix_embeds)
        input_attention_mask.append(attention_mask)
        lora_audio_mask.append(torch.zeros_like(attention_mask))

        if speech_values is not None:
            speech_embeds, speech_attention_mask, _, _, _ = self.get_speech_features(speech_values, speech_attention_mask)
            inputs_embeds.append(speech_embeds)
            input_attention_mask.append(speech_attention_mask)
            lora_audio_mask.append(torch.ones_like(speech_attention_mask))

        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds.append(suffix_embeds)
        input_attention_mask.append(suffix_attention_mask)
        lora_audio_mask.append(torch.zeros_like(suffix_attention_mask))

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        input_attention_mask = torch.cat(input_attention_mask, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)

        self.update_lora_mask(lora_audio_mask, True)

        return self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=input_attention_mask,
            generation_config=generation_config
        )
    
    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config,
        device,
        streamer=None,
    ):
        inputs_embeds = []
        lora_audio_mask = []

        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0].to(device)
                embeds = self.qwen_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
                lora_audio_mask.append(torch.zeros_like(input_ids))
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0].to(device), h[1].to(device)
                speech_embeds, speech_attention_mask, _, _, _= self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
                lora_audio_mask.append(speech_attention_mask)
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)
        self.update_lora_mask(lora_audio_mask, True)

        # context length for decoding
        context_len = inputs_embeds.size(1)

        # provide attention_mask to avoid inference warning when pad token equals eos token
        attention_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device)

        sequences = self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            streamer=streamer,
        )
        return sequences, context_len

