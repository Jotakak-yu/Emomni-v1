import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
from functools import lru_cache
import fire
import soundfile as sf
import mmap
import io

import torch.distributed as dist
import numpy as np
import torch
import random
import datasets
from datasets import Features, Sequence, Value

from dataclasses import dataclass

from transformers import WhisperFeatureExtractor, AutoTokenizer

logger = logging.getLogger(__name__)

emotion2idx = {
    "neutral": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3,
    "surprised": 4
}


# 使用更安全的特征schema定义
def get_feature_schema():
    """动态获取特征schema，提供更好的兼容性"""
    try:
        return Features({
            "start_ids": Sequence(Value("int64")),
            "start_mask": Sequence(Value("int64")),
            "start_labels": Sequence(Value("int64")),
            "instruction_ids": Sequence(Value("int64")),
            "instruction_mask": Sequence(Value("int64")),
            "instruction_labels": Sequence(Value("int64")),
            "audio_instruction_ids": Sequence(Value("int64")),
            "audio_instruction_mask": Sequence(Value("int64")),
            "audio_instruction_labels": Sequence(Value("int64")),
            "input_ids": Sequence(Value("int32")),  # 保持int32以提供更好的兼容性
            "input_mask": Sequence(Value("int64")),
            "input_labels": Sequence(Value("int64")),
            "suffix_ids": Sequence(Value("int64")),
            "suffix_mask": Sequence(Value("int64")),
            "suffix_labels": Sequence(Value("int64")),
            "emotion_labels": Value("int64"),
            "to_keep": Value("bool"),
            "audio_path": Value("string"),
        })
    except Exception as e:
        logger.warning(f"Failed to create feature schema: {e}, using None")
        return None

# 向后兼容的特征schema
feature_schema = get_feature_schema()


def process_dataset(
    batch,
    tokenizer,
    _tokenize_str,
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="input",
    audio_field="audio",
    output_field="output",
    max_length=384,
    min_duration=1.0,
    max_duration=30.0,
    check_audio=True,
    use_emotion=False,
    im_start_tokens=None,
    im_end_tokens=None,
    nl_tokens=None,
    audio_check_sample_rate=1,  # 控制采样检查频率
    _sample_counter=[0],  # 用列表实现可变计数器
):
    """
    构建 ChatML 格式的训练数据，匹配 Qwen2.5/3 官方 chat template。
    
    模型 forward 中有两个并行分支：
    ====================
    
    1. Speech 分支（带音频，用于训练）：
       speech_embeds = [start] + [audio_instruction] + [AUDIO] + [suffix]
       
    2. Text 分支（纯文本，用于 KD teacher）：
       text_embeds = [start] + [instruction] + [input] + [suffix]
    
    对应 ChatML 格式（Speech 分支）：
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {audio_instruction}[AUDIO]<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>
    
    各部分说明：
    - start_ids: "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
    - audio_instruction_ids: 音频**之前**的指令文本（用于 Speech 分支）
    - instruction_ids: 文本指令（用于 Text 分支 KD）
    - [AUDIO]: 音频特征（speech_input_embeds）
    - input_ids: 原始文本（用于 Text 分支 KD，对应 AUDIO 的位置）
    - suffix_ids: "<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n"
    
    instruction vs audio_instruction：
    ====================
    - instruction: 用于 Text 分支（纯文本 KD teacher）
    - audio_instruction: 用于 Speech 分支（带音频）
    - 默认情况下，audio_instruction = instruction（如果未单独指定）
    
    Qwen 版本差异（关键）：
    ====================
    Qwen1:
      - eos_token = <|endoftext|> (151643)
      - 序列结尾: <|im_end|>\n<|endoftext|>
      
    Qwen2.5/3:
      - eos_token = <|im_end|> (151645)  
      - pad_token = <|endoftext|> (151643)
      - 序列结尾: <|im_end|>\n（官方模板，不额外加 <|endoftext|>）
    
    本实现遵循 Qwen2.5/3 官方格式：
    - 序列以 <|im_end|>\n 结尾
    - 不添加额外的 <|endoftext|>
    
    续写任务特殊说明（blsp-emo 风格）：
    ====================
    对于续写任务（cw_labels）的数据格式：
    - text: 原始待续写的文本（如 "the plans you make"）
    - output: 续写内容（**只包含续写部分**，如 "turn out to be garbage."）
      * 注意：output 不重复原 text！
    - audio_instruction: "Continue the following sentence that reflects a '{emotion}' emotion tone..."
    - [AUDIO]: 音频特征（对应原始待续写的文本）
    
    训练数据构建：
    - Text 分支：{instruction}{text} → {output}（续写部分）
    - Speech 分支：{audio_instruction}[AUDIO] → {output}（续写部分）
    
    推理时使用 Assistant Prefill（emotion_text_generation.py）：
    ====================
    输入序列：
      <|im_start|>system\n{system_prompt}<|im_end|>\n
      <|im_start|>user\n{instruction}<|im_end|>\n
      <|im_start|>assistant\n{text}
    
    模型从 {text} 结尾处开始生成续写内容。
    
    这样确保：
    - 生成的 output 只包含续写部分（不含原 text）
    - 与训练数据格式一致（output 字段只有续写）
    - max_new_tokens 不会被原文占用
    """
    if not input_field and not audio_field:
        raise ValueError(f"neither input_field nor audio_field is set for processing batch: {batch}")
    if not output_field:
        raise ValueError(f"output_field not set for processing batch: {batch}")
    if instruction_field:
        instruction = batch[instruction_field]
    if audio_instruction_field:
        audio_instruction = batch[audio_instruction_field]

    # 获取 ChatML 特殊 token
    # Qwen2.5/3 tokenizer 应该包含这些 token
    if im_start_tokens is None or im_end_tokens is None or nl_tokens is None:
        try:
            im_start_tokens = tokenizer.convert_tokens_to_ids(["<|im_start|>"])
            im_end_tokens = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
            nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Failed to get ChatML tokens: {e}, using empty lists")
            im_start_tokens = []
            im_end_tokens = []
            nl_tokens = tokenizer.encode("\n") if hasattr(tokenizer, "encode") else []

    # 根据是否使用情感选择系统提示
    if use_emotion:
        system_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
    else:
        system_prompt = "You are a helpful assistant."

    # ========== 构建 start_ids ==========
    # 格式: <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n
    start_ids = []
    start_ids += im_start_tokens + _tokenize_str(role="system", content=f"{system_prompt}") + im_end_tokens
    start_ids += nl_tokens
    start_ids += im_start_tokens + _tokenize_str(role="user")
    start_mask = [1] * len(start_ids)
    start_labels = [-100] * len(start_ids)  # 不计算 loss

    # ========== 构建 instruction_ids ==========
    # 音频前的指令文本
    instruction_ids, instruction_mask, instruction_labels = [], [], []
    if instruction:
        instruction_ids = _tokenize_str(content=instruction)
        instruction_mask = [1] * len(instruction_ids)
        instruction_labels = [-100] * len(instruction_ids)  # 不计算 loss

    # ========== 构建 audio_instruction_ids ==========
    # 音频后的指令文本（如果有）
    audio_instruction_ids, audio_instruction_mask, audio_instruction_labels = instruction_ids, instruction_mask, instruction_labels
    if audio_instruction:
        audio_instruction_ids = _tokenize_str(content=audio_instruction)
        audio_instruction_mask = [1] * len(audio_instruction_ids)
        audio_instruction_labels = [-100] * len(audio_instruction_ids)

    # ========== 构建 input_ids ==========
    # 原始输入文本（如续写任务的原文）
    input_ids, input_mask, input_labels = [], [], []
    if input_field:
        input_ids = _tokenize_str(content=batch[input_field])
        input_mask = [1] * len(input_ids)
        input_labels = [-100] * len(input_ids)  # 不计算 loss

    # ========== 处理音频 ==========
    audio_path = ""
    to_keep = True
    if audio_field:
        audio_path = batch[audio_field]
        if check_audio:
            # 采样检查：只检查部分样本以减少磁盘 IO
            _sample_counter[0] += 1
            should_check = (_sample_counter[0] % audio_check_sample_rate) == 0
            
            if should_check:
                try:
                    duration = get_audio_duration(audio_path)
                    if duration < min_duration or duration > max_duration:
                        to_keep = False
                except Exception as e:
                    print(f"Error processing audio {audio_path}: {e}")
                    to_keep = False

    # ========== 构建 suffix_ids ==========
    # 格式: <|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n
    suffix_ids, suffix_mask, suffix_labels = [], [], []
    
    # 首先添加 user 消息的结束和 assistant 的开始
    new_ids = im_end_tokens + nl_tokens + im_start_tokens + _tokenize_str(role="assistant")
    suffix_ids += new_ids
    suffix_mask += [1] * len(new_ids)
    suffix_labels += [-100] * len(new_ids)  # 不计算 loss

    # 然后添加 output 内容
    early_stop = batch.get("early_stop", False)
    if early_stop:
        # early_stop=True: 生成被截断，不添加结束 token
        new_ids = _tokenize_str(content=batch[output_field])
    else:
        # 正常结束: 添加 <|im_end|>\n 作为序列结尾
        # 
        # 重要：Qwen2.5/3 使用 <|im_end|> 作为 EOS token
        # 不需要额外添加 <|endoftext|>（那是 Qwen1 的做法）
        # 
        # 官方 chat template 的结尾格式：
        # {assistant_content}<|im_end|>\n (如果是最后一轮对话)
        end_sequence = im_end_tokens + nl_tokens
        new_ids = _tokenize_str(content=batch[output_field]) + end_sequence
    
    suffix_ids += new_ids
    suffix_mask += [1] * len(new_ids)
    suffix_labels += new_ids  # output 部分计算 loss

    # ========== 长度检查 ==========
    if (len(start_ids) + len(instruction_ids) + len(input_ids) + len(suffix_ids)) > max_length:
        to_keep = False
    
    # ========== 处理 emotion 标签 ==========
    emotion_labels = None
    if use_emotion:
        emotion = batch.get("emotion")
        if emotion in emotion2idx:
            emotion_labels = emotion2idx[emotion]
        else:
            # 需要情感标签却缺失时，过滤该样本
            to_keep = False
            if emotion is not None:
                logger.debug(f"Unknown emotion label '{emotion}' encountered; dropping sample.")

    # ========== 保存结果 ==========
    batch["start_ids"] = start_ids
    batch["start_mask"] = start_mask
    batch["start_labels"] = start_labels
    batch["instruction_ids"] = instruction_ids
    batch["instruction_mask"] = instruction_mask
    batch["instruction_labels"] = instruction_labels
    batch["audio_instruction_ids"] = audio_instruction_ids
    batch["audio_instruction_mask"] = audio_instruction_mask
    batch["audio_instruction_labels"] = audio_instruction_labels
    batch["input_ids"] = input_ids
    batch["input_mask"] = input_mask
    batch["input_labels"] = input_labels
    batch["suffix_ids"] = suffix_ids
    batch["suffix_mask"] = suffix_mask
    batch["suffix_labels"] = suffix_labels
    batch["emotion_labels"] = emotion_labels

    batch["to_keep"] = to_keep
    if audio_path:
        batch["audio_path"] = audio_path

    return batch

def load_instruction_dataset(
    manifest_dir="",
    manifest_files="",
    tokenizer=None,
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="",
    audio_field="",
    output_field="",
    max_length=384,
    min_duration=1.0,
    max_duration=30.0,
    num_proc=8,
    use_emotion=False,
    audio_check_sample_rate=1,  # 新增：控制音频校验采样率，1=全部检查，10=每10个检查1个
):
    if not manifest_files:
        logger.warning(f"loading processed dataset from {manifest_dir}")
        dataset = datasets.load_from_disk(manifest_dir)
        return dataset
    
    logger.warning(f"load dataset from scratch from {manifest_dir}/{manifest_files}")
    
    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        manifest_dir, data_files=manifest_files_list, split="train", streaming=False
    )

    try:
        im_start_tokens = tokenizer.convert_tokens_to_ids(["<|im_start|>"])
        im_end_tokens = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
    except Exception as e:
        logger.warning(f"Failed to get ChatML tokens at dataset load: {e}, using empty lists")
        im_start_tokens = []
        im_end_tokens = []
        nl_tokens = tokenizer.encode("\n") if hasattr(tokenizer, "encode") else []


    def _tokenize_str(role="", content=""):
        """
        将 role 和 content 分别 tokenize 并组合。
        
        格式说明：
        - 如果只有 role: 返回 "{role}\n" 的 token
        - 如果只有 content: 返回 "{content}" 的 token
        - 如果两者都有: 返回 "{role}\n{content}" 的 token
        
        这个函数模拟 Qwen1 的 allowed_special=set() 行为，
        使用 add_special_tokens=False 来避免自动添加 BOS/EOS token。
        
        注意：ChatML 的特殊标签（<|im_start|>、<|im_end|>）由调用方单独添加，
        不在这个函数中处理。
        """
        tokens = []
        if role:
            # Role 后面加换行符
            tokens += tokenizer.encode(role, add_special_tokens=False) + tokenizer.encode("\n", add_special_tokens=False)
        if content:
            tokens += tokenizer.encode(content, add_special_tokens=False)
        return tokens
    print(f"Raw dataset size: {len(raw_dataset)}")
    print(f"Raw dataset columns: {raw_dataset.column_names}")
    if len(raw_dataset) > 0:
        print(f"Sample raw data: {raw_dataset[0]}")

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "_tokenize_str": _tokenize_str,
            "instruction": instruction,
            "instruction_field": instruction_field,
            "audio_instruction": audio_instruction,
            "audio_instruction_field": audio_instruction_field,
            "input_field": input_field,
            "audio_field": audio_field,
            "output_field": output_field,
            "max_length": max_length,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "use_emotion": use_emotion,
            "im_start_tokens": im_start_tokens,
            "im_end_tokens": im_end_tokens,
            "nl_tokens": nl_tokens,
            "audio_check_sample_rate": audio_check_sample_rate,
        },
        features=feature_schema if feature_schema is not None else None,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc,
        writer_batch_size=256,  # 增大写入批次，减少磁盘 IO 次数
    )

    print(f"Dataset size before filtering: {len(dataset)}")
    
    def to_keep(flag):
        return flag

    dataset = dataset.filter(
        to_keep,
        input_columns=["to_keep"]
    )
    
    print(f"Dataset size after filtering: {len(dataset)}")
    
    return dataset


def load_instruction_datasets(data_args, tokenizer=None, num_proc=8):
    if os.path.exists(data_args.dataset_save_dir) and os.listdir(data_args.dataset_save_dir):
        logger.warning(f"loading processed dataset from {data_args.dataset_save_dir}")
        try:
            dataset = datasets.load_from_disk(data_args.dataset_save_dir)
            return dataset
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to load processed dataset from {data_args.dataset_save_dir}: {e}")
            logger.warning("This might be due to incompatible feature schema. Falling back to reprocessing data...")
            # 备份原目录并重新处理数据
            import shutil
            backup_dir = data_args.dataset_save_dir + "_backup"
            if not os.path.exists(backup_dir):
                shutil.move(data_args.dataset_save_dir, backup_dir)
                logger.info(f"Backed up incompatible dataset to {backup_dir}")
            else:
                # 如果备份已存在，直接删除当前的
                shutil.rmtree(data_args.dataset_save_dir)
                logger.info(f"Removed incompatible dataset from {data_args.dataset_save_dir}")

    manifest_keys = ["manifest_dirs", "manifest_files", "instructions", "instruction_fields",
                     "audio_instructions", "audio_instruction_fields", "input_fields",
                     "audio_fields", "output_fields"]
    if data_args.dataset_dirs:
        dataset_dirs = data_args.dataset_dirs.split("|")
        all_datasets = [load_instruction_dataset(manifest_dir=dataset_dir) for dataset_dir in dataset_dirs]
        num_datasets = len(all_datasets)
    else:
        manifest_values = [
            (getattr(data_args, key) or "").split("|")  # 添加 or "" 防止 None.split
            for key in manifest_keys
        ]
        num_datasets = len(manifest_values[0])
        if num_datasets == 0:
            raise ValueError("no datasets specified")
        for i, key in enumerate(manifest_keys):
            if len(manifest_values[i]) != num_datasets:
                raise ValueError(f"unexpected number of {key} in {data_args}")
        all_datasets = [load_instruction_dataset(manifest_dir=manifest_values[0][i],
                                                 manifest_files=manifest_values[1][i],
                                                 instruction=manifest_values[2][i],
                                                 instruction_field=manifest_values[3][i],
                                                 audio_instruction=manifest_values[4][i],
                                                 audio_instruction_field=manifest_values[5][i],
                                                 input_field=manifest_values[6][i],
                                                 audio_field=manifest_values[7][i],
                                                 output_field=manifest_values[8][i],
                                                 tokenizer=tokenizer,
                                                 num_proc=num_proc)
                        for i in range(num_datasets)]
    if len(all_datasets) == 1:
        dataset = all_datasets[0]
    else:
        sample_probs = data_args.sample_probs.split("|")
        if len(sample_probs) == num_datasets:
            sample_probs = [float(prob) for prob in sample_probs]
        else:
            if data_args.sample_probs == "None":
                sample_probs = None
            else:
                raise ValueError(f"unexpected number of probabilities in {data_args}")
        dataset = datasets.interleave_datasets(all_datasets, stopping_strategy=data_args.interleave_stopping_strategy,
                                               probabilities=sample_probs)

    
    if data_args.dataset_save_dir and (not dist.is_initialized() or dist.get_rank() == 0):
        dataset.save_to_disk(data_args.dataset_save_dir)

    return dataset

def collate_tokens(
    values: List[List[int]],
    pad_id: int,
    left_pad: bool = False
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        if left_pad:
            copy_tensor(torch.LongTensor(v), res[i][-len(v): ])
        else:
            copy_tensor(torch.LongTensor(v), res[i][: len(v)])

    return res

def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            data = mmap_o[offset : offset + length]
    return data


def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)

def is_sf_audio_data(data: bytes) -> bool:
    is_wav = data[0] == 82 and data[1] == 73 and data[2] == 70
    is_flac = data[0] == 102 and data[1] == 76 and data[2] == 97
    is_ogg = data[0] == 79 and data[1] == 103 and data[2] == 103
    return is_wav or is_flac or is_ogg


def get_waveform(
    path_or_fp: str,
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
    meta = path_or_fp.split(":")
    if len(meta) == 3:
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])
    else:
        path_or_fp = path_or_fp
    
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC/OGG audios")

    ext = Path(path_or_fp).suffix.lower()
    # Try reading with soundfile for formats it handles well
    waveform = None
    sample_rate = None
    try:
        if ext in [".wav", ".flac", ".ogg"]:
            waveform, sample_rate = sf.read(
                path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
            )
        elif ext in [".zip"]:
            data = read_from_stored_zip(path_or_fp, start, frames)
            assert is_sf_audio_data(data)
            f = io.BytesIO(data)
            waveform, sample_rate = sf.read(
                f, dtype="float32", always_2d=True
            )
    except Exception as e:
        logger.warning(f"soundfile read failed for {path_or_fp}: {e}")
        waveform = None
        sample_rate = None

    # Fallback to torchaudio for compressed formats (mp3, m4a, opus, webm, etc.) or when soundfile fails
    if waveform is None:
        try:
            import torchaudio
            ta_waveform, sample_rate = torchaudio.load(path_or_fp)
            # torchaudio returns [channels, length]
            waveform = ta_waveform.numpy().T  # to [length, channels]
        except Exception as e:
            # Last resort: try soundfile regardless of extension
            try:
                waveform, sample_rate = sf.read(
                    path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to read audio {path_or_fp}: torchaudio error: {e}; soundfile error: {e2}")

    # Ensure shape [channels, length]
    waveform = waveform.T  # currently [length, channels] -> [channels, length]

    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalization,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )
    if not normalization:
        # If not normalizing, scale to int16 range for backward-compatibility
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform


@lru_cache(maxsize=500000)
def get_audio_duration(path: str) -> float:
    """获取音频时长（秒），尽可能只读取文件头，避免解码整个音频。

    这在离线预处理阶段极大降低 IO 和 CPU 消耗。
    """
    try:
        info = sf.info(path)
        if info.samplerate and info.frames:
            return info.frames / float(info.samplerate)
    except Exception as e:
        logger.debug(f"soundfile.info failed for {path}: {e}")

    try:
        import torchaudio
        tinfo = torchaudio.info(path)
        if tinfo.sample_rate and tinfo.num_frames:
            return tinfo.num_frames / float(tinfo.sample_rate)
    except Exception as e:
        logger.debug(f"torchaudio.info failed for {path}: {e}")

    # Fallback: 最后尝试一次轻量读取，避免无声文件或奇怪编码阻塞。
    try:
        waveform, sr = sf.read(path, dtype="float32", always_2d=True, frames=1)
        return waveform.shape[0] / float(sr) if sr else 0.0
    except Exception as e:
        raise RuntimeError(f"Failed to read audio duration for {path}: {e}")

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


@dataclass
class InstructionDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """加载单个音频文件"""
        try:
            return get_waveform(audio_path, output_sample_rate=self.sampling_rate)
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            # 返回空音频以保持batch完整性
            return np.zeros(self.sampling_rate, dtype=np.float32)

    def __call__(self, samples: List[Dict]):
        start_ids = [sample["start_ids"] for sample in samples]
        start_mask = [sample["start_mask"] for sample in samples]
        start_labels = [sample["start_labels"] for sample in samples]
        instruction_ids = [sample["instruction_ids"] for sample in samples]
        instruction_mask = [sample["instruction_mask"] for sample in samples]
        instruction_labels = [sample["instruction_labels"] for sample in samples]
        audio_instruction_ids = [sample["audio_instruction_ids"] for sample in samples]
        audio_instruction_mask = [sample["audio_instruction_mask"] for sample in samples]
        audio_instruction_labels = [sample["audio_instruction_labels"] for sample in samples]
        input_ids = [sample["input_ids"] for sample in samples]
        input_mask = [sample["input_mask"] for sample in samples]
        input_labels = [sample["input_labels"] for sample in samples]
        suffix_ids = [sample["suffix_ids"] for sample in samples]
        suffix_mask = [sample["suffix_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]
        emotion_labels = [sample.get("emotion_labels") for sample in samples]

        start_ids = collate_tokens(start_ids, self.pad_id)
        start_mask = collate_tokens(start_mask, 0)
        start_labels = collate_tokens(start_labels, -100)
        instruction_ids = collate_tokens(instruction_ids, self.pad_id)
        instruction_mask = collate_tokens(instruction_mask, 0)
        instruction_labels = collate_tokens(instruction_labels, -100)
        audio_instruction_ids = collate_tokens(audio_instruction_ids, self.pad_id)
        audio_instruction_mask = collate_tokens(audio_instruction_mask, 0)
        audio_instruction_labels = collate_tokens(audio_instruction_labels, -100)
        input_ids = collate_tokens(input_ids, self.pad_id)
        input_mask = collate_tokens(input_mask, 0)
        input_labels = collate_tokens(input_labels, -100)
        suffix_ids = collate_tokens(suffix_ids, self.pad_id)
        suffix_mask = collate_tokens(suffix_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)
        if all(label is None for label in emotion_labels):
            emotion_labels = None
        else:
            if any(label is None for label in emotion_labels):
                missing_indices = [idx for idx, label in enumerate(emotion_labels) if label is None]
                logger.warning(
                    "Mixed emotion label availability detected in collator; skipping emotion loss for samples %s.",
                    missing_indices
                )
                emotion_labels = None
            else:
                emotion_labels = torch.tensor(emotion_labels, dtype=torch.long)

        audio_paths = [sample.get("audio_path", "") for sample in samples]
        has_audio = any(audio_paths)
        
        if not has_audio:
            speech_values = None
            speech_mask = None
        else:
            raw_speech = [
                self._load_audio(path) if path else np.array([], dtype=np.float32)
                for path in audio_paths
            ]
            
            # 过滤掉空音频
            if all(len(sample) == 0 for sample in raw_speech):
                speech_values = None
                speech_mask = None
            else:
                speech_inputs = self.extractor(
                    raw_speech, 
                    sampling_rate=self.sampling_rate, 
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                speech_values = speech_inputs.input_features
                speech_mask = speech_inputs.attention_mask

        return {
            "start_ids": start_ids,
            "start_mask": start_mask,
            "start_labels": start_labels,
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
            "instruction_labels": instruction_labels,
            "audio_instruction_ids": audio_instruction_ids,
            "audio_instruction_mask": audio_instruction_mask,
            "audio_instruction_labels": audio_instruction_labels,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_labels": input_labels,
            "suffix_ids": suffix_ids,
            "suffix_mask": suffix_mask,
            "suffix_labels": suffix_labels,
            "emotion_labels": emotion_labels,
            "speech_values": speech_values,
            "speech_mask": speech_mask
        }


def offline_process(
    dataroot="",
    manifest_files="",
    lm_path="",
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="",
    audio_field="",
    output_field="",
    save_dir="",
    max_length=384,
    min_duration=1.0,
    max_duration=30.0,
    num_proc=8,
    use_emotion=False,
    audio_check_sample_rate=1,  # 新增参数：音频校验采样率（1=全检，10=抽检10%）
):
    text_tokenizer = AutoTokenizer.from_pretrained(
        lm_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 设置特殊token
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # 加载原始数据集以获取过滤前的统计信息
    if manifest_files:
        manifest_files_list = manifest_files.split(",")
        raw_dataset = datasets.load_dataset(
            dataroot, data_files=manifest_files_list, split="train", streaming=False
        )
        raw_size = len(raw_dataset)
    else:
        raw_size = None

    dataset = load_instruction_dataset(
        dataroot,
        manifest_files,
        text_tokenizer,
        instruction,
        instruction_field,
        audio_instruction,
        audio_instruction_field,
        input_field,
        audio_field,
        output_field,
        max_length,
        min_duration,
        max_duration,
        num_proc,
        use_emotion,
        audio_check_sample_rate,
    )
    
    final_size = len(dataset)
    
    # 打印详细的统计信息
    print("\n" + "="*80)
    print("离线处理统计信息")
    print("="*80)
    
    if raw_size is not None:
        filtered_size = raw_size - final_size
        filter_ratio = (filtered_size / raw_size) * 100 if raw_size > 0 else 0.0
        
        print(f"原始数据集大小:           {raw_size}")
        print(f"最终数据集大小:           {final_size}")
        print(f"被筛选掉的样本数量:       {filtered_size}")
        print(f"筛选率:                   {filter_ratio:.2f}%")
        print(f"保留率:                   {100-filter_ratio:.2f}%")
    else:
        print(f"最终数据集大小:           {final_size}")
    
    print("="*80)
    
    if final_size == 0:
        print("⚠️  警告: 处理后数据集为空！")
        return
    
    # 打印数据集配置参数
    print("\n处理配置:")
    print(f"  - 最大序列长度:          {max_length}")
    print(f"  - 音频最小时长:          {min_duration}s")
    print(f"  - 音频最大时长:          {max_duration}s")
    print(f"  - 是否使用情感标签:      {use_emotion}")
    print(f"  - 处理进程数:            {num_proc}")
    print(f"  - 音频校验采样率:        每 {audio_check_sample_rate} 个检查 1 个 ({100.0/audio_check_sample_rate:.1f}%)")
    print()
    
    # 打印第一条样本的数据结构
    print("第一条样本的数据结构:")
    sample = dataset[0]
    
    # 统计token数量
    token_counts = {}
    for key in sample.keys():
        if key not in ["audio_path", "to_keep", "emotion_labels"]:
            token_counts[key] = len(sample[key])
    
    for key in sorted(token_counts.keys()):
        print(f"  - {key:30s}: {token_counts[key]:6d} tokens")
    
    if "audio_path" in sample:
        print(f"  - {'audio_path':30s}: {sample['audio_path']}")
    
    if "emotion_labels" in sample and sample["emotion_labels"] is not None:
        emotion_name = [k for k, v in emotion2idx.items() if v == sample["emotion_labels"]]
        emotion_name = emotion_name[0] if emotion_name else "unknown"
        print(f"  - {'emotion_labels':30s}: {sample['emotion_labels']} ({emotion_name})")
    
    # 统计情感标签分布（如果使用情感标签）
    if use_emotion and "emotion_labels" in dataset.column_names:
        print("\n情感标签分布:")
        emotion_counts = {}
        for sample in dataset:
            label = sample.get("emotion_labels")
            if label is not None:
                emotion_name = [k for k, v in emotion2idx.items() if v == label]
                emotion_name = emotion_name[0] if emotion_name else "unknown"
                emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
        
        for emotion, count in sorted(emotion_counts.items()):
            ratio = (count / final_size) * 100
            print(f"  - {emotion:15s}: {count:6d} ({ratio:6.2f}%)")
    
    print("="*80 + "\n")
    
    if save_dir:
        print(f"保存数据集到: {save_dir}")
        dataset.save_to_disk(save_dir)
        print("✓ 数据集保存成功")


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })
