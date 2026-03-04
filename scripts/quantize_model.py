#!/usr/bin/env python3
"""
Emomni 模型量化脚本
使用 bitsandbytes 对 Emomni 模型进行量化并保存

模型结构分析:
- whisper_model: Whisper Encoder (音频编码器) - 不建议量化，影响音频理解
- adapter: Subsampler/CFormer (适配层) - 不建议量化，参数量小
- qwen_model: Qwen2.5-7B LLM (语言模型) - 主要量化目标
- hidden2emotion: 情感分类头 - 不建议量化，参数量小

BitsAndBytes 量化说明:
- BNB 是动态量化方法，不需要校准集
- 只对 torch.nn.Linear 层进行量化
- 支持 4bit (NF4) 和 8bit (LLM.int8) 量化
- 可以通过 llm_int8_skip_modules 跳过指定模块
"""

import argparse
import os
import sys
import logging
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    WhisperFeatureExtractor
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling_emomni import EmomniModel
from src.configuration_emomni import EmomniConfig

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_quantization_config(
    bits: int = 4,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    skip_modules: list = None
) -> BitsAndBytesConfig:
    """
    创建 BitsAndBytes 量化配置

    Args:
        bits: 量化位数，4 或 8
        compute_dtype: 计算时使用的数据类型
        quant_type: 4bit 量化类型，"nf4" 或 "fp4"
        use_double_quant: 是否使用双重量化（仅4bit）
        skip_modules: 跳过量化的模块名列表

    Returns:
        BitsAndBytesConfig 配置对象
    """
    if skip_modules is None:
        # 默认跳过 lm_head（输出层）以保持精度
        skip_modules = ["lm_head"]

    if bits == 4:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=use_double_quant,
            llm_int8_skip_modules=skip_modules,
        )
        logger.info(f"创建 4-bit 量化配置: quant_type={quant_type}, double_quant={use_double_quant}")
    elif bits == 8:
        config = BitsAndBytesConfig(
            load_in_4bit=False,
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=skip_modules,
        )
        logger.info(f"创建 8-bit 量化配置: threshold=6.0")
    else:
        raise ValueError(f"不支持的量化位数: {bits}，仅支持 4 或 8")

    return config


def quantize_and_save_model(
    model_path: str,
    save_path: str,
    qwen_model: str = None,
    bits: int = 4,
    compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    device: str = "cuda:0",
    merge_lora: bool = True,
):
    """
    量化并保存 Emomni 模型
    
    注意: 对于带有 LoRA 的模型，建议先合并 LoRA 权重再进行量化。
    如果模型有 LoRA 配置且 lora_scope="global"，会自动合并。

    Args:
        model_path: 原始模型路径
        save_path: 量化后模型保存路径
        qwen_model: Qwen 基础模型路径（用于解析缺失的 _name_or_path）
        bits: 量化位数
        compute_dtype: 计算数据类型
        quant_type: 4bit 量化类型
        use_double_quant: 是否使用双重量化
        device: 运行设备
        merge_lora: 是否在量化前合并 LoRA 权重（推荐）
    """
    # 保存原始模型路径（在可能被修改前）
    original_model_path = model_path
    
    # 解析计算数据类型
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype_torch = dtype_map.get(compute_dtype, torch.bfloat16)

    # 创建量化配置
    quantization_config = get_quantization_config(
        bits=bits,
        compute_dtype=compute_dtype_torch,
        quant_type=quant_type,
        use_double_quant=use_double_quant,
        skip_modules=["lm_head"],  # 跳过输出层
    )

    logger.info(f"加载原始模型: {model_path}")

    # 加载 tokenizer 和 feature extractor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Tokenizer 加载完成")

    try:
        extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        logger.info("WhisperFeatureExtractor 加载完成")
    except Exception as e:
        logger.warning(f"无法加载 WhisperFeatureExtractor: {e}")
        extractor = None

    # 加载模型配置
    config = EmomniConfig.from_pretrained(model_path)
    logger.info("模型配置加载完成")

    # 如果提供了 qwen_model 路径，更新配置
    if qwen_model and isinstance(config.qwen_config, dict):
        if not config.qwen_config.get("_name_or_path"):
            config.qwen_config["_name_or_path"] = qwen_model
            logger.info(f"注入 Qwen 模型路径: {qwen_model}")

    # 检查是否需要先加载全精度模型并合并 LoRA
    has_lora = bool(config.lora_config)
    lora_scope = getattr(config, 'lora_scope', 'global')
    
    if has_lora and merge_lora:
        logger.info("检测到 LoRA 配置，将先加载全精度模型并合并 LoRA 权重...")
        
        # 先加载全精度模型 - 使用 device_map 自动放置
        full_model_kwargs = {
            "torch_dtype": compute_dtype_torch,
            "trust_remote_code": True,
            "device_map": device,  # 使用 device_map 而不是 .to(device)
        }
        if qwen_model:
            full_model_kwargs["qwen_model"] = qwen_model
            
        full_model = EmomniModel.from_pretrained(model_path, **full_model_kwargs)
        
        # 合并 LoRA 权重
        if lora_scope == "global":
            logger.info("合并全局 LoRA 权重...")
            full_model.merge_lora()
            logger.info("LoRA 权重合并完成")
        else:
            logger.warning(f"LoRA scope 为 '{lora_scope}'，无法自动合并。将保持 LoRA 结构。")
        
        # 保存合并后的临时模型
        import tempfile
        temp_dir = tempfile.mkdtemp()
        logger.info(f"保存合并后的临时模型到: {temp_dir}")
        full_model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)
        if extractor is not None:
            extractor.save_pretrained(temp_dir)
        
        # 释放全精度模型内存
        del full_model
        torch.cuda.empty_cache()
        
        # 更新加载路径为临时目录
        model_path = temp_dir

    # 加载并量化模型 - 注意不要传入 config，让 from_pretrained 自己加载
    logger.info("开始加载并量化模型...")
    load_kwargs = {
        "device_map": device,
        "quantization_config": quantization_config,
        "trust_remote_code": True,
        "torch_dtype": compute_dtype_torch,
    }
    
    if qwen_model:
        load_kwargs["qwen_model"] = qwen_model

    model = EmomniModel.from_pretrained(model_path, **load_kwargs)
    logger.info("模型量化完成")

    # 打印模型内存占用
    if hasattr(model, 'get_memory_footprint'):
        memory_mb = model.get_memory_footprint() / 1024 / 1024
        logger.info(f"量化后模型内存占用: {memory_mb:.2f} MB")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 保存量化模型
    logger.info(f"保存量化模型到: {save_path}")
    model.save_pretrained(save_path, safe_serialization=True)
    
    # 保存 tokenizer
    tokenizer.save_pretrained(save_path)
    logger.info("Tokenizer 保存完成")

    # 保存 feature extractor
    if extractor is not None:
        extractor.save_pretrained(save_path)
        logger.info("WhisperFeatureExtractor 保存完成")

    # 保存量化信息
    quant_info = {
        "quantization_method": "bitsandbytes",
        "bits": bits,
        "quant_type": quant_type if bits == 4 else "int8",
        "compute_dtype": compute_dtype,
        "use_double_quant": use_double_quant if bits == 4 else False,
        "skip_modules": ["lm_head"],
        "original_model_path": original_model_path,
    }
    
    quant_info_path = os.path.join(save_path, "quantization_config.json")
    with open(quant_info_path, "w", encoding="utf-8") as f:
        json.dump(quant_info, f, indent=2, ensure_ascii=False)
    logger.info(f"量化信息保存到: {quant_info_path}")

    logger.info("=" * 50)
    logger.info("量化完成!")
    logger.info(f"原始模型: {original_model_path}")
    logger.info(f"量化模型: {save_path}")
    logger.info(f"量化配置: {bits}-bit {quant_type if bits == 4 else 'int8'}")
    logger.info("=" * 50)

    return model


def main():
    parser = argparse.ArgumentParser(description="Emomni 模型量化脚本")
    
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="原始模型路径"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="量化后模型保存路径"
    )
    parser.add_argument(
        "--qwen_model", type=str, default=None,
        help="Qwen 基础模型路径（用于解析配置中缺失的 _name_or_path）"
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[4, 8],
        help="量化位数: 4 (NF4/FP4) 或 8 (LLM.int8)"
    )
    parser.add_argument(
        "--compute_dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="计算数据类型"
    )
    parser.add_argument(
        "--quant_type", type=str, default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit 量化类型: nf4 (推荐) 或 fp4"
    )
    parser.add_argument(
        "--no_double_quant", action="store_true",
        help="禁用双重量化（仅 4-bit 有效）"
    )
    parser.add_argument(
        "--no_merge_lora", action="store_true",
        help="禁用 LoRA 权重合并（不推荐）"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="运行设备"
    )

    args = parser.parse_args()

    quantize_and_save_model(
        model_path=args.model_path,
        save_path=args.save_path,
        qwen_model=args.qwen_model,
        bits=args.bits,
        compute_dtype=args.compute_dtype,
        quant_type=args.quant_type,
        use_double_quant=not args.no_double_quant,
        device=args.device,
        merge_lora=not args.no_merge_lora,
    )


if __name__ == "__main__":
    main()