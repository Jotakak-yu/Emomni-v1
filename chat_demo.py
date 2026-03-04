import argparse
import os
import random
import time
import logging
import sys
import re
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from gradio import processing_utils
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
from transformers import TextIteratorStreamer
import threading
import queue
import shutil

from transformers import WhisperFeatureExtractor
from transformers import GenerationConfig, AutoTokenizer
from src.modeling_emomni import EmomniModel
from src.instruction_dataset import get_waveform
from src.qwen_generation_utils import get_stop_words_ids, decode_tokens

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Chat Demo")

class ChatHistory(object):
    def __init__(self, 
        tokenizer, 
        extractor, 
        max_window_size=6144,
        max_new_tokens=512,
        use_emotion=False,
        speech_downsample_rate=16
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.max_window_size = max_window_size
        self.max_new_tokens = max_new_tokens
        self.speech_downsample_rate = speech_downsample_rate

        self.im_start_tokens = tokenizer.convert_tokens_to_ids(["<|im_start|>"])
        self.im_end_tokens = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
        self.nl_tokens = tokenizer.encode("\n")

        ### add system
        if use_emotion:
            sys_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
        else:
            sys_prompt = "You are a helpful assistant."
        input_ids = self.im_start_tokens + self._tokenize_str("system", f"{sys_prompt}") + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.system_histroy = [(input_ids,)]
        self.system_length = input_ids.shape[1]

        self.reset()
    
    def reset(self):
        self.history = []
        self.lengths = []
        self.cur_length = self.system_length
        self.audio_file = []
        self.audio_to_history = True
    
    def _tokenize_str(self, role, content):
        """Enhanced tokenization with better Qwen2.5/3 support."""
        try:
            # Prefer no special tokens for role/content to avoid duplication
            role_tokens = self.tokenizer.encode(role, add_special_tokens=False)
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
            return role_tokens + self.nl_tokens + content_tokens
        except Exception:
            # Fallback to original method
            return self.tokenizer.encode(role, add_special_tokens=True) + self.nl_tokens + self.tokenizer.encode(content, add_special_tokens=True)

    def add_text_history(self, role, text):
        input_ids =  self.nl_tokens + self.im_start_tokens + self._tokenize_str(role, text) + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

    def add_audio(self, audio_file):
        self.audio_to_history = False
        self.audio_file.append(audio_file)

    def add_speech_history(self, speech, text=""):
        if self.audio_to_history:
            return
        self.audio_to_history = True
        speech = get_waveform(speech, output_sample_rate=self.extractor.sampling_rate)
        speech_inputs = self.extractor(
            speech,
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )
        speech_values = speech_inputs.input_features.to(torch.bfloat16)
        speech_attention_mask = speech_inputs.attention_mask

        input_ids = self.nl_tokens + self.im_start_tokens + self._tokenize_str("user", text)
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

        self.history.append(
            (speech_values, speech_attention_mask)
        )
        length = speech_attention_mask.sum().item() // self.speech_downsample_rate
        self.lengths.append(length)
        self.cur_length += length
        

        input_ids = [] + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]
    
    def get_history(self):
        input_ids = self.nl_tokens + self.im_start_tokens + self.tokenizer.encode("assistant")
        input_ids = torch.LongTensor([input_ids])
        length = input_ids.shape[1]

        while self.cur_length > (self.max_window_size - self.max_new_tokens - length):
            pop_length = self.lengths.pop(0)
            self.history.pop(0)
            self.cur_length -= pop_length
        return self.system_histroy + self.history + [(input_ids,)]


def parse_args():
    parser = argparse.ArgumentParser(description="Emomni-Qwen Chat Demo with Enhanced Qwen2.5/3 Support")
    parser.add_argument(
        "--emlm_model", type=str, default=None,
        help="Path to the Emomni model", required=True
    )
    parser.add_argument(
        "--qwen_model", type=str, default=None,
        help="Base Qwen LLM id or path to resolve missing _name_or_path in saved config"
    )
    parser.add_argument(
        "--use_emotion", action="store_true",
        help="Whether to use emotion-aware mode"
    )
    parser.add_argument(
        "--force_chat_format", type=str, choices=["chatml", "raw"], default=None,
        help="Force specific chat format (overrides auto-detection)"
    )
    ### Enhanced generation args for Qwen2.5/3
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="Minimum new tokens for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperature for generation (0.7 recommended for Qwen2.5/3)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.05,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--max_window_size", type=int, default=6144,
        help="Maximum length for previous context"
    )
    # TTS related arguments
    parser.add_argument(
        "--enable_tts", action="store_true",
        help="Enable TTS voice reply (default: disabled)"
    )
    parser.add_argument(
        "--tts_api_url", type=str, default="http://127.0.0.1:8882",
        help="TTS API server URL (default: http://127.0.0.1:8882)"
    )
    parser.add_argument(
        "--tts_role", type=str, default="中文女",
        choices=['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女'],
        help="TTS voice role"
    )
    parser.add_argument(
        "--tts_split_mode", type=str, default="punctuation",
        choices=['punctuation', 'fast'],
        help="TTS sentence splitting mode: 'punctuation' for sentence-level, 'fast' for faster streaming"
    )
    parser.add_argument(
        "--tts_speed", type=float, default=1.0,
        help="TTS speech speed (0.5-2.0)"
    )
    # Quantization arguments
    parser.add_argument(
        "--load_in_4bit", action="store_true",
        help="Load model with 4-bit quantization (NF4)"
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true",
        help="Load model with 8-bit quantization (LLM.int8)"
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for 4-bit quantization"
    )
    parser.add_argument(
        "--bnb_4bit_quant_type", type=str, default="nf4",
        choices=["nf4", "fp4"],
        help="Quantization type for 4-bit (nf4 recommended)"
    )
    parser.add_argument(
        "--no_double_quant", action="store_true",
        help="Disable double quantization for 4-bit"
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()

# Validate quantization args
if args.load_in_4bit and args.load_in_8bit:
    raise ValueError("Cannot use both --load_in_4bit and --load_in_8bit")

accelerator = Accelerator()
logger.info(accelerator.state)

device = accelerator.device

# Enhanced model loading with better Qwen2.5/3 support
try:
    tokenizer = AutoTokenizer.from_pretrained(args.emlm_model, trust_remote_code=True)
    logger.info(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

try:
    extractor = WhisperFeatureExtractor.from_pretrained(args.emlm_model)
    logger.info("Loaded Whisper feature extractor")
except Exception as e:
    logger.error(f"Failed to load feature extractor: {e}")
    raise

# Prepare quantization config if needed
quantization_config = None
dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

if args.load_in_4bit:
    compute_dtype = dtype_map.get(args.bnb_4bit_compute_dtype, torch.bfloat16)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=torch.uint8,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=not args.no_double_quant,
        llm_int8_skip_modules=["lm_head"],
    )
    logger.info(f"Using 4-bit quantization: quant_type={args.bnb_4bit_quant_type}, "
               f"double_quant={not args.no_double_quant}")
elif args.load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head"],
    )
    logger.info("Using 8-bit quantization (LLM.int8)")

# Enhanced model loading with error handling and quantization support
try:
    load_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    
    if quantization_config:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = device
    
    if args.qwen_model:
        load_kwargs["qwen_model"] = args.qwen_model
    
    model = EmomniModel.from_pretrained(args.emlm_model, **load_kwargs)
    
    if args.qwen_model:
        logger.info(f"Loaded model with specified Qwen base: {args.qwen_model}")
    else:
        logger.info("Loaded model with default configuration")
    
    if quantization_config:
        if hasattr(model, 'get_memory_footprint'):
            memory_mb = model.get_memory_footprint() / 1024 / 1024
            logger.info(f"Quantized model memory footprint: {memory_mb:.2f} MB")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

try:
    generation_config = GenerationConfig.from_pretrained(args.emlm_model)
    logger.info("Loaded generation config")
except Exception as e:
    logger.warning(f"Failed to load generation config, using defaults: {e}")
    generation_config = GenerationConfig()

# 增强的chat_format检测，更好支持Qwen2.5/3
def detect_chat_format(tokenizer, generation_config):
    """Enhanced chat format detection for Qwen2.5/3 compatibility."""
    # Check if generation_config already has chat_format set
    if hasattr(generation_config, "chat_format") and generation_config.chat_format:
        return generation_config.chat_format
    
    # Check for built-in chat template (Qwen2.5/3 preferred method)
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        logger.info("Using native chat template from tokenizer")
        return "chatml"
    
    # Check for ChatML special tokens
    try:
        im_start_ids = tokenizer.convert_tokens_to_ids(["<|im_start|>"])
        im_end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
        if (isinstance(im_start_ids, list) and len(im_start_ids) > 0 and im_start_ids[0] is not None and
            isinstance(im_end_ids, list) and len(im_end_ids) > 0 and im_end_ids[0] is not None):
            logger.info("Detected ChatML format via special tokens")
            return "chatml"
    except Exception as e:
        logger.debug(f"ChatML token detection failed: {e}")
    
    # Check tokenizer model name for Qwen variants
    model_name = getattr(tokenizer, "name_or_path", "").lower()
    if any(name in model_name for name in ["qwen", "chatml"]):
        logger.info(f"Detected Qwen model from name: {model_name}")
        return "chatml"
    
    # Default fallback
    logger.info("Using raw format as fallback")
    return "raw"

try:
    generation_config.chat_format = detect_chat_format(tokenizer, generation_config)
except Exception as e:
    logger.warning(f"Chat format detection failed: {e}, using 'raw' as fallback")
    generation_config.chat_format = "raw"

stop_words_ids = get_stop_words_ids(generation_config.chat_format, tokenizer)

# Enhanced generation config for Qwen2.5/3
def setup_generation_config(generation_config, tokenizer, args):
    """Setup generation config with Qwen2.5/3 optimizations."""
    # Set basic parameters
    config_updates = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "max_length": args.max_window_size + args.max_new_tokens,
        "num_return_sequences": 1,
        "do_sample": True,
    }
    
    # Add advanced sampling parameters if specified
    if hasattr(args, 'top_p') and args.top_p:
        config_updates["top_p"] = args.top_p
    if hasattr(args, 'repetition_penalty') and args.repetition_penalty:
        config_updates["repetition_penalty"] = args.repetition_penalty
    
    # Handle special tokens robustly
    try:
        # Try to use newline as BOS for consistency
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
        if nl_tokens:
            config_updates["bos_token_id"] = nl_tokens[0]
    except Exception:
        pass
    
    # Set pad_token_id if not already set
    if getattr(generation_config, "pad_token_id", None) is None:
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            config_updates["pad_token_id"] = tokenizer.pad_token_id
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            config_updates["pad_token_id"] = tokenizer.eos_token_id
    
    # Set eos_token_id if not already set
    if getattr(generation_config, "eos_token_id", None) is None:
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            config_updates["eos_token_id"] = tokenizer.eos_token_id
    
    generation_config.update(**config_updates)
    return generation_config

generation_config = setup_generation_config(generation_config, tokenizer, args)

# Override chat format if specified
if args.force_chat_format:
    generation_config.chat_format = args.force_chat_format
    logger.info(f"Forced chat format to: {args.force_chat_format}")

# Only move to device if not using quantization (quantization handles device placement)
if not quantization_config:
    model = model.to(device)
model.eval()
history = ChatHistory(tokenizer, extractor, generation_config.max_length, generation_config.max_new_tokens, args.use_emotion)

# ========================================
#             TTS Manager (API-based)
# ========================================

class TTSManager:
    """TTS Manager for streaming voice reply generation via HTTP API."""
    
    # Punctuation patterns for sentence splitting
    PUNCT_ZH = r'[。！？；：、]'  # Chinese punctuation
    PUNCT_EN = r'[.!?;:]'  # English punctuation  
    PUNCT_ALL = r'[。！？；：、.!?;:]'  # Combined
    
    # Fast mode: split on shorter phrases
    FAST_SPLIT_ZH = r'[，。！？；：、]'  # Include comma for faster splits
    FAST_SPLIT_EN = r'[,.!?;:]'  # Include comma for faster splits
    FAST_SPLIT_ALL = r'[，。！？；：、,.!?;:]'
    
    VOICE_LIST = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    
    def __init__(self, enabled=False, role='中文女', split_mode='punctuation', speed=1.0, api_url='http://127.0.0.1:8882'):
        self.enabled = enabled
        self.role = role
        self.split_mode = split_mode  # 'punctuation' or 'fast'
        self.speed = speed
        self.api_url = api_url
        
        self.audio_queue = queue.Queue()
        self.tts_thread = None
        self.stop_signal = False
        
        # TTS output directory
        self.output_dir = Path(__file__).parent / "tts_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def split_sentences(self, text, mode='punctuation'):
        """Split text into sentences based on punctuation.
        
        Args:
            text: Input text to split
            mode: 'punctuation' for full sentence, 'fast' for faster shorter splits
            
        Returns:
            Tuple of (sentences_list, remaining_text) or empty list if no complete sentences
        """
        if not text:
            return []
        
        # Choose pattern based on mode
        if mode == 'fast':
            pattern = self.FAST_SPLIT_ALL
        else:
            pattern = self.PUNCT_ALL
        
        sentences = []
        remaining = text
        
        # Find sentences ending with punctuation
        matches = list(re.finditer(pattern, remaining))
        
        if not matches:
            return []
        
        start = 0
        for match in matches:
            end = match.end()
            sentence = remaining[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
        
        # Return sentences and remaining text
        remaining_text = remaining[start:].strip() if start < len(remaining) else ""
        return sentences, remaining_text
    
    def synthesize(self, text):
        """Synthesize speech from text via TTS API.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to generated audio file or None
        """
        if not self.enabled or not text.strip():
            return None
        
        try:
            import requests
            
            # Call TTS API
            response = requests.post(
                f"{self.api_url}/tts",
                data={
                    "text": text,
                    "role": self.role,
                    "speed": self.speed
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"TTS API error: {response.status_code} - {response.text}")
                return None
            
            # Save audio to file
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            filename = f"tts-{timestamp}-{random.randint(1000, 9999)}.wav"
            output_path = self.output_dir / filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"TTS audio saved: {output_path}")
            return str(output_path)
            
        except requests.exceptions.ConnectionError:
            logger.error(f"TTS API connection failed. Make sure tts_api.py is running at {self.api_url}")
            return None
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None
    
    def start_streaming_tts(self, text_queue):
        """Start streaming TTS in background thread.
        
        Args:
            text_queue: Queue to receive text chunks from generation
        """
        if not self.enabled:
            return
        
        self.stop_signal = False
        self.audio_queue = queue.Queue()
        
        def _tts_worker():
            buffer = ""
            while not self.stop_signal:
                try:
                    text_chunk = text_queue.get(timeout=0.1)
                    if text_chunk is None:  # End signal
                        # Synthesize remaining buffer
                        if buffer.strip():
                            audio_path = self.synthesize(buffer)
                            if audio_path:
                                self.audio_queue.put(audio_path)
                        self.audio_queue.put(None)  # Signal end
                        break
                    
                    buffer += text_chunk
                    
                    # Try to split and synthesize complete sentences
                    result = self.split_sentences(buffer, self.split_mode)
                    if result:
                        sentences, remaining = result
                        for sentence in sentences:
                            audio_path = self.synthesize(sentence)
                            if audio_path:
                                self.audio_queue.put(audio_path)
                        buffer = remaining
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"TTS worker error: {e}")
                    break
        
        self.tts_thread = threading.Thread(target=_tts_worker)
        self.tts_thread.daemon = True
        self.tts_thread.start()
    
    def stop_streaming_tts(self):
        """Stop streaming TTS."""
        self.stop_signal = True
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)
    
    def get_audio(self, timeout=0.1):
        """Get next audio file from queue.
        
        Returns:
            Path to audio file, None if queue empty or end signal received
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return "empty"  # Distinguish from None (end signal)
    
    def concat_audio_files(self, audio_files, output_filename=None):
        """Concatenate multiple audio files into one.
        
        Args:
            audio_files: List of audio file paths to concatenate
            output_filename: Optional output filename, auto-generated if None
            
        Returns:
            Path to concatenated audio file or None if failed
        """
        if not audio_files:
            return None
        
        if len(audio_files) == 1:
            return audio_files[0]
        
        try:
            import subprocess
            
            # Generate output filename
            if output_filename is None:
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                output_filename = f"tts-concat-{timestamp}-{random.randint(1000, 9999)}.wav"
            
            output_path = self.output_dir / output_filename
            
            # Create file list for ffmpeg
            list_file = self.output_dir / f"concat_list_{random.randint(1000, 9999)}.txt"
            with open(list_file, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(list_file),
                '-c', 'copy',
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60
            )
            
            # Clean up list file
            if list_file.exists():
                list_file.unlink()
            
            if result.returncode == 0 and output_path.exists():
                logger.debug(f"Concatenated {len(audio_files)} audio files to {output_path}")
                return str(output_path)
            else:
                logger.error(f"FFmpeg concat failed: {result.stderr.decode()}")
                # Fallback: return the last audio file
                return audio_files[-1] if audio_files else None
                
        except Exception as e:
            logger.error(f"Audio concatenation failed: {e}")
            # Fallback: return the last audio file
            return audio_files[-1] if audio_files else None
    
    def cleanup_old_files(self, max_age_hours=24):
        """Clean up old TTS audio files."""
        try:
            cutoff = time.time() - (max_age_hours * 3600)
            for f in self.output_dir.glob("tts-*.wav"):
                if f.stat().st_mtime < cutoff:
                    f.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup TTS files: {e}")


# Initialize TTS Manager
tts_manager = TTSManager(
    enabled=args.enable_tts,
    role=args.tts_role,
    split_mode=args.tts_split_mode,
    speed=args.tts_speed,
    api_url=args.tts_api_url
)

print('Initialization Finished')


def gradio_reset():
    history.reset()
    return None, gr.update(value="", interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), None


def gradio_answer(chatbot, num_beams, temperature, enable_tts, tts_role, tts_split_mode, tts_speed):
    generation_config.update(
        **{
            "num_beams": num_beams, 
            "temperature": temperature,
        }
    )
    
    # Set stop_words_ids in generation_config if it exists
    if stop_words_ids:
        generation_config.stop_words_ids = stop_words_ids

    # Prepare streamer for real-time generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    thread_error = {"msg": None}
    
    # Update TTS settings dynamically
    tts_manager.enabled = enable_tts
    tts_manager.role = tts_role
    tts_manager.split_mode = tts_split_mode
    tts_manager.speed = tts_speed
    
    # Queue for TTS text chunks
    tts_text_queue = queue.Queue() if enable_tts else None

    # Kick off generation in a background thread
    def _run_generation():
        try:
            model.chat(
                history=history.get_history(),
                generation_config=generation_config,
                device=device,
                streamer=streamer,
            )
        except Exception as e:
            thread_error["msg"] = str(e)
            logging.exception("Generation thread failed")

    thread = threading.Thread(target=_run_generation)
    thread.daemon = True
    thread.start()
    
    # Start TTS streaming if enabled
    if enable_tts and tts_text_queue:
        tts_manager.start_streaming_tts(tts_text_queue)

    # Initialize assistant message and stream updates
    chatbot[-1][1] = ""
    cumulative_text = ""
    audio_files = []  # Collect all audio files for concatenation
    first_audio_played = False  # Track if first audio has been sent to player

    # Stream tokens with queue if available; otherwise fallback to iterator
    if hasattr(streamer, "text_queue"):
        while thread.is_alive() or not streamer.text_queue.empty():
            try:
                new_text = streamer.text_queue.get(timeout=0.2)
            except Exception:
                new_text = None
            if new_text:
                cumulative_text += new_text
                chatbot[-1][1] = cumulative_text
                
                # Send text to TTS queue
                if enable_tts and tts_text_queue:
                    tts_text_queue.put(new_text)
                
                # Collect available audio
                if enable_tts:
                    while True:
                        audio = tts_manager.get_audio(timeout=0.01)
                        if audio == "empty":
                            break
                        elif audio is None:
                            break
                        else:
                            audio_files.append(audio)
                    
                    # Play first audio immediately for quick response
                    if audio_files and not first_audio_played:
                        first_audio_played = True
                        yield chatbot, audio_files[0]
                    else:
                        yield chatbot, gr.update()  # Don't update audio to avoid interruption
                else:
                    yield chatbot, None
            if thread_error["msg"] is not None:
                break
    else:
        for new_text in streamer:
            if new_text:
                cumulative_text += new_text
                chatbot[-1][1] = cumulative_text
                
                # Send text to TTS queue
                if enable_tts and tts_text_queue:
                    tts_text_queue.put(new_text)
                
                # Collect available audio
                if enable_tts:
                    while True:
                        audio = tts_manager.get_audio(timeout=0.01)
                        if audio == "empty":
                            break
                        elif audio is None:
                            break
                        else:
                            audio_files.append(audio)
                    
                    # Play first audio immediately for quick response
                    if audio_files and not first_audio_played:
                        first_audio_played = True
                        yield chatbot, audio_files[0]
                    else:
                        yield chatbot, gr.update()  # Don't update audio to avoid interruption
                else:
                    yield chatbot, None
            if thread_error["msg"] is not None:
                break

    # Signal end of text generation to TTS
    if enable_tts and tts_text_queue:
        tts_text_queue.put(None)
    
    # Wait for remaining audio and collect all
    final_audio = None
    if enable_tts:
        while True:
            audio = tts_manager.get_audio(timeout=0.5)
            if audio == "empty":
                # Check if TTS thread is still alive
                if tts_manager.tts_thread and tts_manager.tts_thread.is_alive():
                    continue
                else:
                    break
            elif audio is None:
                break
            else:
                audio_files.append(audio)
        
        # Concatenate all audio files for complete playback
        if audio_files:
            if len(audio_files) == 1:
                final_audio = audio_files[0]
            else:
                final_audio = tts_manager.concat_audio_files(audio_files)

    # Finalize
    if thread_error["msg"] is not None and not cumulative_text:
        chatbot[-1][1] = f"[Error] {thread_error['msg']}"
        yield chatbot, None
        return

    final_response = cumulative_text.strip()
    history.add_text_history("assistant", final_response)
    
    # Return final concatenated audio (user can replay full response)
    yield chatbot, final_audio


title = """<h1 align="center">Demo of Emomni-Qwen</h1>"""
description = """<h3>This is the demo of Emomni-Qwen. Upload your audios and start chatting!</h3>"""
article = """<p><a href='https://xxx.github.io'><img src='https://xxx'></a></p><p><a href='https://github.com/xxx'><img src='https://xxx'></a></p><p><a href='xxx'><img src='xxx'></a></p>
"""


#TODO show examples below


def add_text(chatbot, user_message):
    chatbot = chatbot + [(user_message, None)]
    history.add_text_history("user", user_message)
    return chatbot, gr.update(value="", interactive=False)


def add_file(chatbot, gr_audio):
    try:
        # Resolve source path from gradio file-like object
        src_path = getattr(gr_audio, "name", gr_audio)
        if not src_path or not os.path.exists(src_path):
            raise FileNotFoundError(f"Uploaded file not found: {src_path}")

        # Persist to a stable uploads directory
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        filename = os.path.basename(src_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lower() or ".wav"
        dst_path = os.path.join(
            uploads_dir,
            f"upload_{int(time.time())}_{random.randint(1000,9999)}{ext}"
        )
        shutil.copy2(src_path, dst_path)

        history.add_audio(dst_path)
        history.add_speech_history(history.audio_file[-1])
        chatbot = chatbot + [((dst_path,), None)]
    except Exception as e:
        print(e)
    return chatbot


def add_micophone_file(chatbot, gr_audio_mic):
    if gr_audio_mic is not None:
        try:
            audio = processing_utils.audio_from_file(gr_audio_mic)
            sample_rate, audio_data = audio[0], audio[1]

            # Ensure shape [num_samples, num_channels]
            if audio_data.ndim == 1:
                write_data = audio_data
            elif audio_data.ndim == 2:
                # If shape is [channels, samples], transpose to [samples, channels]
                if audio_data.shape[0] < audio_data.shape[1]:
                    write_data = audio_data.T
                else:
                    write_data = audio_data
            else:
                write_data = np.squeeze(audio_data)

            # Persist microphone recording to stable uploads directory
            uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            dst_path = os.path.join(
                uploads_dir,
                f"mic_{int(time.time())}_{random.randint(1000,9999)}.wav"
            )
            processing_utils.audio_to_file(sample_rate, write_data, dst_path)

            history.add_audio(dst_path)
            history.add_speech_history(history.audio_file[-1])
            chatbot = chatbot + [((dst_path,), None)]
        except Exception as e:
            logging.exception(e)
    return chatbot, gr.update(value=None, interactive=True)


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)

    chatbot = gr.Chatbot([], elem_id="chatbot", height=600, avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))))
    
    # TTS Audio output
    tts_audio_output = gr.Audio(
        label="🔊 TTS Voice Reply",
        type="filepath",
        autoplay=True,
        visible=True,
    )

    with gr.Row():
        num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam",
            )
            
        temperature = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temp",
            )
    
    # TTS Control Row
    with gr.Row():
        enable_tts = gr.Checkbox(
            label="🔊 Enable TTS Voice Reply",
            value=args.enable_tts,
            interactive=True,
        )
        tts_role = gr.Dropdown(
            label="Voice Role",
            choices=['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女'],
            value=args.tts_role,
            interactive=True,
        )
        tts_split_mode = gr.Radio(
            label="Split Mode",
            choices=['punctuation', 'fast'],
            value=args.tts_split_mode,
            interactive=True,
        )
        tts_speed = gr.Slider(
            minimum=0.5,
            maximum=2.0,
            value=args.tts_speed,
            step=0.1,
            interactive=True,
            label="TTS Speed",
        )
    
    with gr.Row():
        clear = gr.Button("🔄 Restart")
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter, or upload an audio",
            container=False)
        btn = gr.UploadButton("📁", file_types=["video", "audio"])
        input_audio_mic = gr.Audio(
            label="🎤",
            type="numpy",
            sources=["microphone", "upload"],
            visible=True,
        )

    # Event handlers with TTS support
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        gradio_answer, 
        [chatbot, num_beams, temperature, enable_tts, tts_role, tts_split_mode, tts_speed], 
        [chatbot, tts_audio_output]
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        gradio_answer, 
        [chatbot, num_beams, temperature, enable_tts, tts_role, tts_split_mode, tts_speed], 
        [chatbot, tts_audio_output]
    )

    input_audio_mic.change(add_micophone_file, [chatbot, input_audio_mic], [chatbot, input_audio_mic], queue=False).then(
        gradio_answer, 
        [chatbot, num_beams, temperature, enable_tts, tts_role, tts_split_mode, tts_speed], 
        [chatbot, tts_audio_output]
    )
    clear.click(gradio_reset, [], [chatbot, txt, input_audio_mic, btn, tts_audio_output], queue=False)

demo.queue()
demo.launch(share=False, server_name="0.0.0.0", server_port=7861)