import os
import argparse
import json
from tqdm import tqdm
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import WhisperFeatureExtractor
from transformers import GenerationConfig, AutoTokenizer
from src.modeling_emomni import EmomniModel
from src.instruction_dataset import get_waveform
from src.qwen_generation_utils import get_stop_words_ids
import logging

logger = logging.getLogger(__name__)


def collate_tokens(values: List[List[int]], pad_id: int):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)
    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)
    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][-len(v):])
    return res


@dataclass
class DataCollator:
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()
    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        reference = [sample["reference"] for sample in samples]
        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)
        raw_speech = [
            get_waveform(sample["audio"], output_sample_rate=self.sampling_rate) if sample["audio"] is not None else []
            for sample in samples
        ]
        if all(len(sample) == 0 for sample in raw_speech):
            speech_values = None
            speech_attention_mask = None
        else:
            speech_inputs = self.extractor(
                raw_speech, 
                sampling_rate=self.sampling_rate, 
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_values = speech_inputs.input_features.to(torch.bfloat16)
            speech_attention_mask = speech_inputs.attention_mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "speech_values": speech_values,
            "speech_attention_mask": speech_attention_mask,
            "reference": reference
        }


def render_chat(tokenizer, messages: List[Dict], add_generation_prompt: bool = False, 
                omit_last_end: bool = False) -> str:
    """
    Render chat messages in ChatML format.
    
    优先使用 tokenizer 的官方 apply_chat_template（Qwen2.5/3 推荐）。
    仅当 tokenizer 没有 chat_template 时才使用手动格式化。
    
    Args:
        tokenizer: The tokenizer
        messages: List of message dicts with 'role' and 'content'
        add_generation_prompt: If True, add '<|im_start|>assistant\n' at the end
        omit_last_end: If True, don't add '<|im_end|>\n' after the last message
                       (used for audio insertion point)
    
    Returns:
        Formatted chat string
    """
    # 优先使用 tokenizer 的官方 chat_template
    if hasattr(tokenizer, 'apply_chat_template') and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            if omit_last_end:
                # 需要在 user 消息后插入音频，不加结束标签
                # apply_chat_template 不直接支持这个，需要手动处理
                pass  # 继续使用手动格式化
            else:
                # 正常情况：使用官方模板
                result = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=add_generation_prompt
                )
                return result
        except Exception as e:
            logger.warning(f"apply_chat_template failed: {e}, falling back to manual formatting")
    
    # 手动 ChatML 格式化（回退方案，或 omit_last_end=True 时使用）
    pieces = []
    for i, m in enumerate(messages):
        role = m.get("role", "user")
        content = m.get("content", "")
        
        # Start the message
        piece = f"<|im_start|>{role}\n{content}"
        
        # Add end tag unless this is the last message and omit_last_end is True
        is_last = (i == len(messages) - 1)
        if not (is_last and omit_last_end):
            piece += "<|im_end|>"
        
        pieces.append(piece)
    
    result = "\n".join(pieces)
    
    # Add generation prompt if requested
    if add_generation_prompt:
        # If we omitted the last end tag, we need to add it now before the generation prompt
        # This ensures the sequence is: ...user content<|im_end|>\n<|im_start|>assistant\n
        if omit_last_end:
            result += "<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Last message already has <|im_end|>, just add newline and generation prompt
            result += "\n<|im_start|>assistant\n"
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--emlm_model", type=str, required=True, help="Path to the blsp model")
    parser.add_argument("--instruction", type=str, default="", help="the general instruction for each example")
    parser.add_argument("--audio_field", type=str, default="", help="the audio filed for each example")
    parser.add_argument("--text_field", type=str, default="", help="the text field for each example")
    parser.add_argument("--reference_field", type=str, default="", help="the reference field for each example")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--use_emotion", type=bool, default=False, help="use emotion sensitive system message")
    # generation args
    parser.add_argument("--max_new_tokens", type=int, default=128, help="max new tokens for generation")
    parser.add_argument("--min_new_tokens", type=int, default=1, help="min new tokens for generation")
    parser.add_argument("--do_sample", action="store_true", help="enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p for generation")
    parser.add_argument("--top_k", type=int, default=0, help="top_k for generation")
    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.emlm_model)
    extractor = WhisperFeatureExtractor.from_pretrained(args.emlm_model)
    generation_config = GenerationConfig.from_pretrained(args.emlm_model)

    # Enhanced special token normalization for Qwen2.5/3
    def normalize_special_tokens(tokenizer, generation_config):
        """Robust special token normalization for different Qwen versions."""
        # Handle pad_token
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                try:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.debug("Set pad_token to eos_token")
                except Exception as e:
                    logger.debug(f"Failed to set pad_token: {e}")
            # Alternative: try using a common pad token
            elif hasattr(tokenizer, 'convert_tokens_to_ids'):
                try:
                    pad_id = tokenizer.convert_tokens_to_ids(["<|pad|>"])
                    if pad_id and pad_id[0] is not None:
                        tokenizer.pad_token_id = pad_id[0]
                        logger.debug("Set pad_token_id from <|pad|> token")
                except Exception:
                    pass
        
        # Sync generation_config with tokenizer
        if getattr(generation_config, "pad_token_id", None) is None:
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                generation_config.pad_token_id = tokenizer.pad_token_id
        
        if getattr(generation_config, "eos_token_id", None) is None:
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Handle bos_token_id for consistency
        if getattr(generation_config, "bos_token_id", None) is None:
            if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
                generation_config.bos_token_id = tokenizer.bos_token_id
        
        return tokenizer, generation_config
    
    tokenizer, generation_config = normalize_special_tokens(tokenizer, generation_config)

    # Enhanced chat_format detection and stop_words setup
    def detect_and_setup_chat_format(tokenizer, generation_config):
        """Enhanced chat format detection for Qwen2.5/3."""
        # Check if already set
        if hasattr(generation_config, "chat_format") and generation_config.chat_format:
            chat_format = generation_config.chat_format
        else:
            # Auto-detect based on tokenizer capabilities
            chat_format = "raw"  # default
            
            # Check for built-in chat template (Qwen2.5/3 preferred)
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                chat_format = "chatml"
                logger.info("Detected chat template, using chatml format")
            else:
                # Check for ChatML special tokens
                try:
                    im_start_ids = tokenizer.convert_tokens_to_ids(["<|im_start|>"])
                    im_end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
                    if (isinstance(im_start_ids, list) and len(im_start_ids) > 0 and im_start_ids[0] is not None and
                        isinstance(im_end_ids, list) and len(im_end_ids) > 0 and im_end_ids[0] is not None):
                        chat_format = "chatml"
                        logger.info("Detected ChatML tokens, using chatml format")
                except Exception:
                    pass
                
                # Check model name
                if chat_format == "raw":
                    model_name = getattr(tokenizer, "name_or_path", "").lower()
                    if "qwen" in model_name:
                        chat_format = "chatml"
                        logger.info(f"Detected Qwen model from name: {model_name}")
            
            generation_config.chat_format = chat_format
        
        # Setup stop words
        try:
            stop_words_ids = get_stop_words_ids(generation_config.chat_format, tokenizer)
            if stop_words_ids:
                generation_config.stop_words_ids = stop_words_ids
                logger.debug(f"Set stop_words_ids for {generation_config.chat_format} format")
        except Exception as e:
            logger.warning(f"Failed to setup stop words: {e}")
        
        return generation_config
    
    generation_config = detect_and_setup_chat_format(tokenizer, generation_config)

    with open(args.input_file, "r") as fin:
        lines = fin.readlines()
        lines = [json.loads(line.strip()) for line in lines]

    dataset = Dataset.from_list(lines)

    def process_dataset(batch):
        """
        构建 SER（语音情感识别）推理的输入序列。
        
        模型 generate 方法中的序列拼接：
        ====================
        inputs_embeds = [prefix_embeds] + [speech_embeds] + [suffix_embeds]
        
        对应训练时的 Speech 分支：
        speech_embeds = [start] + [audio_instruction] + [AUDIO] + [suffix]
        
        推理时的映射：
        - input_ids (prefix) → start + audio_instruction（音频前）
        - speech_embeds → [AUDIO]
        - suffix_input_ids → suffix（音频后 + generation prompt）
        
        ChatML 格式：
        ====================
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {instruction}[AUDIO]{text_input}<|im_end|>
        <|im_start|>assistant
        {生成内容}
        
        SER 任务说明：
        ====================
        - instruction (audio_instruction): 音频**之前**的指令
          例如："Please identify the emotion tone of the speech provided below...Speech: "
        - [AUDIO]: 音频特征位置
        - text_input: 音频**之后**的文本（SER 任务通常为空）
        
        关键：omit_last_end=True
        ====================
        在 user 消息结束前插入音频，因此 user 消息不能有结束标签。
        音频插入后再由 suffix 补上 <|im_end|>\n<|im_start|>assistant\n
        """
        # 获取指令（可以从参数或数据中获取）
        instruction = args.instruction if args.instruction else ""
        
        # text_field 用于在音频后添加文本（对应训练的 input_ids）
        # 对于 SER 任务通常为空
        text_input = ""
        if args.text_field:
            text_input = batch.get(args.text_field, "")

        # System prompt 必须与训练格式完全一致！
        system_prompt = (
            "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
            if args.use_emotion
            else "You are a helpful assistant."
        )

        # 构建 user 消息内容
        # instruction 是音频之前的指令
        user_content = instruction
        
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 构建 input_ids（prefix，对应 start + audio_instruction）
        # 关键：omit_last_end=True，因为音频要插入到 user 消息末尾
        # 格式：<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}
        input_text_no_prompt = render_chat(tokenizer, input_messages, 
                                           add_generation_prompt=False, 
                                           omit_last_end=True)
        try:
            input_ids_no_prompt = tokenizer.encode(input_text_no_prompt, add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Encoding failed with add_special_tokens=False, trying with True: {e}")
            input_ids_no_prompt = tokenizer.encode(input_text_no_prompt, add_special_tokens=True)

        # 构建 suffix（音频后的部分 + generation prompt）
        # 格式：{text_input}<|im_end|>\n<|im_start|>assistant\n
        # 
        # 如果有 text_input（音频后的文本），需要先添加它
        # 然后是 <|im_end|>\n<|im_start|>assistant\n
        suffix_parts = []
        if text_input:
            suffix_parts.append(text_input)
        suffix_parts.append("<|im_end|>\n<|im_start|>assistant\n")
        suffix_text = "".join(suffix_parts)
        
        try:
            suffix_input_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        except Exception:
            suffix_input_ids = tokenizer.encode(suffix_text, add_special_tokens=True)

        # 保存结果
        batch["input_ids"] = input_ids_no_prompt
        batch["attention_mask"] = [1] * len(batch["input_ids"])
        batch["suffix_input_ids"] = suffix_input_ids
        batch["suffix_attention_mask"] = [1] * len(batch["suffix_input_ids"])

        # 音频路径
        batch["audio"] = batch.get(args.audio_field, None) if args.audio_field else None

        # 参考答案
        batch["reference"] = batch.get(args.reference_field, "") if args.reference_field else ""
        return batch

    dataset = dataset.map(process_dataset)

    model = EmomniModel.from_pretrained(args.emlm_model, torch_dtype=torch.bfloat16)

    data_collator = DataCollator(generation_config.pad_token_id, extractor.sampling_rate, extractor)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.batch_size)

    generation_config.update(
        **{
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_beams": 1,
            "num_return_sequences": 1,
        }
    )

    model = model.cuda()
    model.eval()

    with open(args.output_file, "w") as fout:
        for batch in tqdm(dataloader):
            outputs = model.generate(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                suffix_input_ids=batch["suffix_input_ids"].cuda(),
                suffix_attention_mask=batch["suffix_attention_mask"].cuda(),
                speech_values=batch["speech_values"].cuda() if batch["speech_values"] is not None else None,
                speech_attention_mask=batch["speech_attention_mask"].cuda() if batch["speech_attention_mask"] is not None else None,
                generation_config=generation_config,
            )
            output_text = [
                tokenizer.decode(output, skip_special_tokens=True, errors='replace')
                for output in outputs
            ]
            for reference, response in zip(batch["reference"], output_text):
                json_string = json.dumps({"response": response, "reference": reference}, ensure_ascii=False)
                print(json_string, file=fout, flush=True)


if __name__ == "__main__":
    main()
