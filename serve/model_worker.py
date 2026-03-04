"""
Emomni Model Worker
Executes model inference and provides streaming generation.
Based on FastChat model worker architecture.
"""

import argparse
import asyncio
import json
import time
import threading
import uuid
from functools import partial
from typing import List, Optional, Generator

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn

from transformers import (
    AutoTokenizer,
    WhisperFeatureExtractor,
    GenerationConfig,
    TextIteratorStreamer,
    BitsAndBytesConfig
)

from .constants import (
    WORKER_HEART_BEAT_INTERVAL,
    DEFAULT_WORKER_PORT,
    DEFAULT_CONTROLLER_PORT,
    SERVER_ERROR_MSG,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P
)
from .utils import build_logger, pretty_print_semaphore, get_model_display_name

# Generate unique worker ID
worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

# Global state
global_counter = 0
model_semaphore = None


def heart_beat_worker(worker: "ModelWorker"):
    """Background thread to send heartbeats to controller."""
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        worker.send_heart_beat()


class ChatHistory:
    """Manages conversation history for multi-turn chat."""
    
    def __init__(
        self, 
        tokenizer, 
        extractor,
        max_window_size: int = 6144,
        max_new_tokens: int = 512,
        use_emotion: bool = False,
        speech_downsample_rate: int = 16
    ):
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.max_window_size = max_window_size
        self.max_new_tokens = max_new_tokens
        self.speech_downsample_rate = speech_downsample_rate

        # Special tokens
        self.im_start_tokens = tokenizer.convert_tokens_to_ids(["<|im_start|>"])
        self.im_end_tokens = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
        self.nl_tokens = tokenizer.encode("\n")

        # System prompt
        if use_emotion:
            sys_prompt = "You are a helpful assistant. Your name is Emo-Omni, developed by ZJU ISUC Lab. Your response should fulfill requests with empathy toward user's emotion tone."
        else:
            sys_prompt = "You are a helpful assistant. Your name is Emo-Omni, developed by ZJU ISUC Lab."
        
        input_ids = self.im_start_tokens + self._tokenize_str("system", sys_prompt) + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        # Fixed: use system_histroy to match chat_demo.py (typo preserved for compatibility)
        self.system_histroy = [(input_ids,)]
        self.system_length = input_ids.shape[1]

        self.reset()
    
    def reset(self):
        """Reset conversation history."""
        self.history = []
        self.lengths = []
        self.cur_length = self.system_length
        self.audio_file = []
        self.audio_to_history = True
    
    def _tokenize_str(self, role: str, content: str) -> List[int]:
        """Tokenize a role-content pair."""
        try:
            role_tokens = self.tokenizer.encode(role, add_special_tokens=False)
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
            return role_tokens + self.nl_tokens + content_tokens
        except Exception:
            return self.tokenizer.encode(role, add_special_tokens=True) + \
                   self.nl_tokens + self.tokenizer.encode(content, add_special_tokens=True)

    def add_text_history(self, role: str, text: str):
        """Add a text message to history."""
        input_ids = self.nl_tokens + self.im_start_tokens + self._tokenize_str(role, text) + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append((input_ids,))
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

    def add_audio(self, audio_file: str):
        """Mark that an audio file will be added."""
        self.audio_to_history = False
        self.audio_file.append(audio_file)

    def add_speech_history(self, speech_path: str, text: str = ""):
        """Add speech features to history."""
        if self.audio_to_history:
            return
        self.audio_to_history = True
        
        # Import here to avoid circular imports
        from src.instruction_dataset import get_waveform
        
        speech = get_waveform(speech_path, output_sample_rate=self.extractor.sampling_rate)
        speech_inputs = self.extractor(
            speech,
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )
        speech_values = speech_inputs.input_features.to(torch.bfloat16)
        speech_attention_mask = speech_inputs.attention_mask

        # User turn start
        input_ids = self.nl_tokens + self.im_start_tokens + self._tokenize_str("user", text)
        input_ids = torch.LongTensor([input_ids])
        self.history.append((input_ids,))
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

        # Speech features
        self.history.append((speech_values, speech_attention_mask))
        length = speech_attention_mask.sum().item() // self.speech_downsample_rate
        self.lengths.append(length)
        self.cur_length += length

        # User turn end - match chat_demo.py exactly with [] + 
        input_ids = [] + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append((input_ids,))
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]
    
    def get_history(self) -> List:
        """Get history with assistant prompt appended."""
        # Match chat_demo.py exactly: encode "assistant" with default behavior
        input_ids = self.nl_tokens + self.im_start_tokens + self.tokenizer.encode("assistant")
        input_ids = torch.LongTensor([input_ids])
        length = input_ids.shape[1]

        # Trim old history if needed
        while self.cur_length > (self.max_window_size - self.max_new_tokens - length):
            if not self.lengths:
                break
            pop_length = self.lengths.pop(0)
            self.history.pop(0)
            self.cur_length -= pop_length
        
        # Fixed: use system_histroy to match initialization
        return self.system_histroy + self.history + [(input_ids,)]


class ModelWorker:
    """
    Model worker that handles inference requests.
    
    Features:
    - Emomni model loading and inference
    - Streaming generation
    - Heartbeat to controller
    - Multi-turn conversation support
    - BitsAndBytes quantization support (4-bit/8-bit)
    """

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        device: str = "cuda",
        no_register: bool = False,
        use_emotion: bool = False,
        qwen_model: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.device = device
        self.use_emotion = use_emotion
        
        # Quantization settings
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        self._load_model(model_path, qwen_model)
        
        # Create display name (last two path components)
        self.model_name = get_model_display_name(model_path)
        if self.load_in_4bit:
            self.model_name += " (4-bit)"
        elif self.load_in_8bit:
            self.model_name += " (8-bit)"
        logger.info(f"Model loaded with display name: {self.model_name}")

        # Register with controller
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, 
                args=(self,), 
                daemon=True
            )
            self.heart_beat_thread.start()

    def _load_model(self, model_path: str, qwen_model: Optional[str] = None):
        """Load the Emomni model and related components with optional quantization."""
        from src.modeling_emomni import EmomniModel
        from src.qwen_generation_utils import get_stop_words_ids
        
        # Load tokenizer and extractor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        
        # Prepare quantization config if needed
        quantization_config = None
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        if self.load_in_4bit:
            compute_dtype = dtype_map.get(self.bnb_4bit_compute_dtype, torch.bfloat16)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_storage=torch.uint8,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                llm_int8_skip_modules=["lm_head"],
            )
            logger.info(f"Using 4-bit quantization: quant_type={self.bnb_4bit_quant_type}, "
                       f"double_quant={self.bnb_4bit_use_double_quant}")
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=False,
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"],
            )
            logger.info("Using 8-bit quantization (LLM.int8)")
        
        # Load model with or without quantization
        load_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = self.device
        
        if qwen_model:
            load_kwargs["qwen_model"] = qwen_model
        
        self.model = EmomniModel.from_pretrained(model_path, **load_kwargs)
        
        # Only move to device if not using quantization (quantization handles device placement)
        if not quantization_config:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Log memory usage
        if hasattr(self.model, 'get_memory_footprint'):
            memory_mb = self.model.get_memory_footprint() / 1024 / 1024
            logger.info(f"Model memory footprint: {memory_mb:.2f} MB")
        
        # Load generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception:
            self.generation_config = GenerationConfig()
        
        # Detect and set chat format (critical for proper generation)
        self._detect_chat_format()
        
        # Setup generation config
        self._setup_generation_config()
        
        # Get stop words based on chat format
        chat_format = getattr(self.generation_config, "chat_format", "chatml")
        self.stop_words_ids = get_stop_words_ids(chat_format, self.tokenizer)
        
        logger.info(f"Chat format: {chat_format}")
        logger.info("Model loading complete")

    def _detect_chat_format(self):
        """Enhanced chat format detection for Qwen2.5/3 compatibility."""
        # Check if generation_config already has chat_format set
        if hasattr(self.generation_config, "chat_format") and self.generation_config.chat_format:
            return
        
        # Check for built-in chat template (Qwen2.5/3 preferred method)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            logger.info("Using native chat template from tokenizer")
            self.generation_config.chat_format = "chatml"
            return
        
        # Check for ChatML special tokens
        try:
            im_start_ids = self.tokenizer.convert_tokens_to_ids(["<|im_start|>"])
            im_end_ids = self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])
            if (isinstance(im_start_ids, list) and len(im_start_ids) > 0 and im_start_ids[0] is not None and
                isinstance(im_end_ids, list) and len(im_end_ids) > 0 and im_end_ids[0] is not None):
                logger.info("Detected ChatML format via special tokens")
                self.generation_config.chat_format = "chatml"
                return
        except Exception as e:
            logger.debug(f"ChatML token detection failed: {e}")
        
        # Check tokenizer model name for Qwen variants
        model_name = getattr(self.tokenizer, "name_or_path", "").lower()
        if any(name in model_name for name in ["qwen", "chatml"]):
            logger.info(f"Detected Qwen model from name: {model_name}")
            self.generation_config.chat_format = "chatml"
            return
        
        # Default fallback
        logger.info("Using raw format as fallback")
        self.generation_config.chat_format = "raw"

    def _setup_generation_config(self):
        """Configure generation parameters with Qwen2.5/3 optimizations."""
        # Calculate max_length based on max_window_size (matching chat_demo.py)
        max_window_size = 6144  # Default from chat_demo.py
        
        config_updates = {
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "min_new_tokens": 1,  # Add min_new_tokens like chat_demo.py
            "temperature": DEFAULT_TEMPERATURE,
            "max_length": max_window_size + DEFAULT_MAX_NEW_TOKENS,  # Like chat_demo.py
            "top_p": DEFAULT_TOP_P,
            "do_sample": True,
            "num_return_sequences": 1,
        }
        
        # Try to use newline as BOS for consistency
        try:
            nl_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
            if nl_tokens:
                config_updates["bos_token_id"] = nl_tokens[0]
        except Exception:
            pass
        
        # Set pad_token_id
        if getattr(self.generation_config, "pad_token_id", None) is None:
            if self.tokenizer.pad_token_id is not None:
                config_updates["pad_token_id"] = self.tokenizer.pad_token_id
            elif self.tokenizer.eos_token_id is not None:
                config_updates["pad_token_id"] = self.tokenizer.eos_token_id
        
        # Set eos_token_id if not already set
        if getattr(self.generation_config, "eos_token_id", None) is None:
            if self.tokenizer.eos_token_id is not None:
                config_updates["eos_token_id"] = self.tokenizer.eos_token_id
        
        self.generation_config.update(**config_updates)

    def register_to_controller(self):
        """Register this worker with the controller."""
        logger.info(f"Registering with controller: {self.controller_addr}")
        
        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        
        try:
            r = requests.post(url, json=data, timeout=10)
            if r.status_code == 200:
                logger.info("Successfully registered with controller")
            else:
                logger.error(f"Failed to register: {r.status_code}")
        except Exception as e:
            logger.error(f"Failed to register with controller: {e}")

    def send_heart_beat(self):
        """Send heartbeat to controller."""
        logger.debug(f"Sending heartbeat. Model: {self.model_name}")
        
        url = self.controller_addr + "/receive_heart_beat"
        
        try:
            ret = requests.post(url, json={
                "worker_name": self.worker_addr,
                "queue_length": self.get_queue_length()
            }, timeout=5)
            
            exist = ret.json().get("exist", True)
            if not exist:
                logger.warning("Controller lost our registration, re-registering...")
                self.register_to_controller()
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")

    def get_queue_length(self) -> int:
        """Get current queue length."""
        global model_semaphore
        if model_semaphore is None:
            return 0
        return args.limit_model_concurrency - model_semaphore._value + \
               (len(model_semaphore._waiters) if model_semaphore._waiters else 0)

    def get_status(self) -> dict:
        """Get worker status."""
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def create_chat_history(self, use_emotion: bool = None) -> ChatHistory:
        """Create a new chat history instance."""
        if use_emotion is None:
            use_emotion = self.use_emotion
        return ChatHistory(
            self.tokenizer,
            self.extractor,
            max_window_size=getattr(self.generation_config, "max_length", 6144),
            max_new_tokens=getattr(self.generation_config, "max_new_tokens", 512),
            use_emotion=use_emotion
        )

    @torch.inference_mode()
    def generate_stream(self, params: dict) -> Generator[bytes, None, None]:
        """Generate response with streaming."""
        try:
            # Extract parameters
            prompt = params.get("prompt", "")
            audio_path = params.get("audio_path")
            temperature = float(params.get("temperature", DEFAULT_TEMPERATURE))
            top_p = float(params.get("top_p", DEFAULT_TOP_P))
            max_new_tokens = int(params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
            use_emotion = params.get("use_emotion", self.use_emotion)
            
            # Create chat history
            history = self.create_chat_history(use_emotion)
            
            # Process input
            if audio_path:
                history.add_audio(audio_path)
                history.add_speech_history(audio_path, prompt)
            else:
                history.add_text_history("user", prompt)
            
            # Update generation config
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0.001 else 0.001,
                top_p=top_p,
                do_sample=temperature > 0.001,
                pad_token_id=self.generation_config.pad_token_id,
                eos_token_id=self.generation_config.eos_token_id,
            )
            if self.stop_words_ids:
                gen_config.stop_words_ids = self.stop_words_ids
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True,
                timeout=60
            )
            
            # Start generation in background thread
            def generate_thread():
                try:
                    self.model.chat(
                        history=history.get_history(),
                        generation_config=gen_config,
                        device=self.device,
                        streamer=streamer,
                    )
                except Exception as e:
                    logger.error(f"Generation error: {e}")
            
            thread = threading.Thread(target=generate_thread)
            thread.daemon = True
            thread.start()
            
            # Stream results
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield json.dumps({
                    "text": generated_text,
                    "error_code": 0,
                    "finish_reason": None
                }).encode() + b"\0"
            
            # Final response
            history.add_text_history("assistant", generated_text)
            yield json.dumps({
                "text": generated_text,
                "error_code": 0,
                "finish_reason": "stop"
            }).encode() + b"\0"
            
        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            yield json.dumps({
                "text": SERVER_ERROR_MSG,
                "error_code": 1,
            }).encode() + b"\0"

    def generate_stream_gate(self, params: dict) -> Generator[bytes, None, None]:
        """Wrapper for generate_stream with error handling."""
        try:
            for x in self.generate_stream(params):
                yield x
        except Exception as e:
            logger.exception(f"Stream generation error: {e}")
            yield json.dumps({
                "text": SERVER_ERROR_MSG,
                "error_code": 1,
            }).encode() + b"\0"


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI()
worker: ModelWorker = None
args = None


def release_model_semaphore(fn=None):
    """Release semaphore and optionally call a function."""
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    """Handle streaming generation request."""
    global model_semaphore, global_counter
    global_counter += 1
    
    params = await request.json()
    
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    
    await model_semaphore.acquire()
    worker.send_heart_beat()
    
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(
        partial(release_model_semaphore, fn=worker.send_heart_beat)
    )
    
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    """Get worker status."""
    return worker.get_status()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": worker.model_name,
        "worker_id": worker.worker_id
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emomni Model Worker")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_WORKER_PORT)
    parser.add_argument(
        "--worker-address", 
        type=str,
        default=f"http://localhost:{DEFAULT_WORKER_PORT}"
    )
    parser.add_argument(
        "--controller-address", 
        type=str,
        default=f"http://localhost:{DEFAULT_CONTROLLER_PORT}"
    )
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the Emomni model")
    parser.add_argument("--qwen-model", type=str, default=None,
                       help="Base Qwen model path (optional)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--use-emotion", action="store_true",
                       help="Enable emotion-aware mode by default")
    # Quantization arguments
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model with 4-bit quantization (NF4)")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model with 8-bit quantization (LLM.int8)")
    parser.add_argument("--bnb-4bit-compute-dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Compute dtype for 4-bit quantization")
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4",
                       choices=["nf4", "fp4"],
                       help="Quantization type for 4-bit (nf4 recommended)")
    parser.add_argument("--no-double-quant", action="store_true",
                       help="Disable double quantization for 4-bit")
    args = parser.parse_args()
    
    # Validate quantization args
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Cannot use both --load-in-4bit and --load-in-8bit")
    
    logger.info(f"Starting model worker with args: {args}")
    
    worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_path=args.model_path,
        device=args.device,
        no_register=args.no_register,
        use_emotion=args.use_emotion,
        qwen_model=args.qwen_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=not args.no_double_quant,
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
