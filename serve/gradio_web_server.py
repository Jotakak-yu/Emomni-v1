"""
Emomni Gradio Web Server
Provides a web UI for interacting with Emomni models.
Features streaming generation and TTS voice reply.
"""

import argparse
import datetime
import json
import os
import time
import uuid
import re
import queue
import threading
import shutil
import random
from pathlib import Path
from typing import Optional, List, Tuple, Generator

import gradio as gr
import requests
import numpy as np
from gradio import processing_utils

from openai import OpenAI
from .constants import (
    LOGDIR,
    DEFAULT_WEBUI_PORT,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_TTS_API_URL,
    DEFAULT_TTS_ROLE,
    TTS_VOICE_LIST,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    SERVER_ERROR_MSG,
    MODERATION_MSG
)
from .utils import build_logger, violates_moderation, get_model_display_name

logger = build_logger("gradio_web_server", "gradio_web_server.log")

# HTTP headers
headers = {"User-Agent": "Emomni Client"}

# Button state helpers
no_change_btn = gr.update()
enable_btn = gr.update(interactive=True)
disable_btn = gr.update(interactive=False)


# ============================================================
# TTS Manager
# ============================================================

class TTSManager:
    """TTS Manager for streaming voice reply generation via HTTP API."""
    
    PUNCT_ALL = r'[。！？；：、.!?;:]'
    FAST_SPLIT_ALL = r'[，。！？；：、,.!?;:]'
    
    def __init__(
        self, 
        enabled: bool = True,
        role: str = DEFAULT_TTS_ROLE,
        split_mode: str = "punctuation",
        speed: float = 1.0,
        api_url: str = DEFAULT_TTS_API_URL
    ):
        self.enabled = enabled
        self.role = role
        self.split_mode = split_mode
        self.speed = speed
        self.api_url = api_url.rstrip("/")

        self.model = "tts-1"
        self.client = OpenAI(
            api_key=os.getenv("TTS_API_KEY") or os.getenv("OPENAI_API_KEY") or "12314",
            base_url=self.api_url,
        )
        
        self.audio_queue = queue.Queue()
        self.tts_thread = None
        self.stop_signal = False
        
        # Output directory
        self.output_dir = Path(LOGDIR) / "tts_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def split_sentences(self, text: str, mode: str = "punctuation") -> Tuple[List[str], str]:
        """Split text into sentences based on punctuation."""
        if not text:
            return [], ""
        
        pattern = self.FAST_SPLIT_ALL if mode == "fast" else self.PUNCT_ALL
        sentences = []
        remaining = text
        
        matches = list(re.finditer(pattern, remaining))
        if not matches:
            return [], text
        
        start = 0
        for match in matches:
            end = match.end()
            sentence = remaining[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
        
        remaining_text = remaining[start:].strip() if start < len(remaining) else ""
        return sentences, remaining_text
    
    def synthesize(self, text: str, timeout: int = 60) -> Optional[str]:
        """Synthesize speech from text via TTS API (OpenAI-compatible)."""
        if not self.enabled or not text.strip():
            return None
        
        voice = self.role
        if isinstance(voice, dict):
            voice = (
                voice.get("value")
                or voice.get("id")
                or voice.get("name")
                or str(voice)
            )
        
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        filename = f"tts-{timestamp}-{random.randint(1000, 9999)}.wav"
        output_path = self.output_dir / filename

        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=voice,
                input=text,
                speed=self.speed,
            ) as response:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

            return str(output_path)

        except Exception as e:
            logger.error(f"TTS synthesize failed: {e}")
            return None

    
    def start_streaming_tts(self, text_queue: queue.Queue):
        """Start streaming TTS in background thread."""
        if not self.enabled:
            return
        
        self.stop_signal = False
        self.audio_queue = queue.Queue()
        
        def _tts_worker():
            buffer = ""
            while not self.stop_signal:
                try:
                    text_chunk = text_queue.get(timeout=0.1)
                    if text_chunk is None:
                        if buffer.strip():
                            audio_path = self.synthesize(buffer)
                            if audio_path:
                                self.audio_queue.put(audio_path)
                        self.audio_queue.put(None)
                        break
                    
                    buffer += text_chunk
                    sentences, remaining = self.split_sentences(buffer, self.split_mode)
                    if sentences:
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
    
    def get_audio(self, timeout: float = 0.1):
        """Get next audio file from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return "empty"
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds, or 0 if failed
        """
        try:
            import wave
            with wave.open(audio_path, 'r') as audio_file:
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception:
            # Some formats (e.g., mp3) can't be read by wave; try ffprobe for accurate duration
            try:
                import subprocess
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(audio_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    duration = float(result.stdout.strip())
                    if duration > 0:
                        return duration
            except Exception as e:
                logger.debug(f"ffprobe failed for {audio_path}: {e}")
            
            # Fallback: estimate based on file size (rough estimate)
            try:
                file_size = os.path.getsize(audio_path)
                estimated_duration = file_size / 32000.0  # 16kHz 16-bit mono WAV ≈ 32KB/sec
                return max(0.5, estimated_duration)
            except Exception as e:
                logger.warning(f"Failed to estimate audio duration for {audio_path}: {e}")
                return 1.0  # Default fallback
    
    def concat_audio_files(self, audio_files: List[str]) -> Optional[str]:
        """Concatenate multiple audio files into one."""
        if not audio_files:
            return None
        if len(audio_files) == 1:
            return audio_files[0]
        
        try:
            import subprocess
            
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            output_path = self.output_dir / f"tts-concat-{timestamp}.wav"
            
            list_file = self.output_dir / f"concat_list_{random.randint(1000, 9999)}.txt"
            with open(list_file, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            result = subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                 '-i', str(list_file), '-c', 'copy', str(output_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
            )
            
            if list_file.exists():
                list_file.unlink()
            
            if result.returncode == 0 and output_path.exists():
                return str(output_path)
            return audio_files[-1]
            
        except Exception as e:
            logger.error(f"Audio concatenation failed: {e}")
            return audio_files[-1] if audio_files else None


# Global TTS manager
tts_manager = TTSManager()

def warmup_tts_service(manager: TTSManager, text: str = "你好"):
    """Send a lightweight request to warm up / validate the TTS service."""
    if not manager.enabled:
        logger.info("TTS is disabled, skip warmup check.")
        return
    try:
        logger.info("Running TTS warmup request...")
        test_audio = manager.synthesize(text, timeout=15)
        if test_audio:
            logger.info("TTS warmup succeeded.")
            try:
                Path(test_audio).unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.debug(f"Failed to remove TTS warmup file {test_audio}: {cleanup_err}")
        else:
            logger.warning("TTS warmup failed: no audio returned.")
    except Exception as e:
        logger.warning(f"TTS warmup error: {e}")


# ============================================================
# Conversation State
# ============================================================

class ConversationState:
    """Manages conversation state for the web UI."""
    
    def __init__(self):
        self.messages = []  # List of (role, content) tuples
        self.audio_files = []  # List of audio file paths
        self.skip_next = False
    
    def reset(self):
        """Reset conversation state."""
        self.messages = []
        self.audio_files = []
        self.skip_next = False
    
    def add_message(self, role: str, content):
        """Add a message to the conversation."""
        self.messages.append([role, content])
    
    def to_chatbot_format(self) -> List:
        """Convert messages to Gradio chatbot format."""
        chatbot = []
        for role, content in self.messages:
            if role == "user":
                if isinstance(content, tuple):
                    # Audio message
                    chatbot.append([content, None])
                else:
                    chatbot.append([content, None])
            elif role == "assistant":
                if chatbot and chatbot[-1][1] is None:
                    chatbot[-1][1] = content
                else:
                    chatbot.append([None, content])
        return chatbot
    
    def copy(self):
        """Create a copy of the conversation state."""
        new_state = ConversationState()
        new_state.messages = [list(m) for m in self.messages]
        new_state.audio_files = list(self.audio_files)
        new_state.skip_next = self.skip_next
        return new_state


def get_conv_log_filename() -> str:
    """Get filename for conversation log."""
    t = datetime.datetime.now()
    return os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")


# ============================================================
# Model API Functions
# ============================================================

def get_model_list(controller_url: str) -> List[str]:
    """Get list of available models from controller."""
    try:
        ret = requests.post(controller_url + "/refresh_all_workers", timeout=5)
        ret = requests.post(controller_url + "/list_models", timeout=5)
        if ret.status_code == 200:
            models = ret.json().get("models", [])
            models.sort()
            logger.info(f"Available models: {models}")
            return models
    except Exception as e:
        logger.error(f"Failed to get model list: {e}")
    return []


def get_worker_address(controller_url: str, model_name: str) -> str:
    """Get worker address for a model."""
    try:
        ret = requests.post(
            controller_url + "/get_worker_address",
            json={"model": model_name},
            timeout=5
        )
        if ret.status_code == 200:
            return ret.json().get("address", "")
    except Exception as e:
        logger.error(f"Failed to get worker address: {e}")
    return ""


# ============================================================
# Gradio Event Handlers
# ============================================================

def clear_history(request: gr.Request):
    """Clear conversation history."""
    logger.info(f"clear_history. ip: {request.client.host}")
    state = ConversationState()
    # Return enable_btn so buttons can be clicked again after clearing
    return (state, [], "", None) + (enable_btn,) * 2 + (None,)


def add_text(state, text, audio_input, use_emotion, request: gr.Request):
    """Add text message to conversation."""
    logger.info(f"add_text. ip: {request.client.host}, text_len: {len(text)}")
    
    if state is None or not isinstance(state, ConversationState):
        state = ConversationState()
    
    # Input validation
    if len(text) <= 0 and audio_input is None:
        state.skip_next = True
        return (state, state.to_chatbot_format(), "", None) + (no_change_btn,) * 2
    
    # Content moderation
    if violates_moderation(text):
        state.skip_next = True
        return (state, state.to_chatbot_format(), MODERATION_MSG, None) + (no_change_btn,) * 2
    
    # Handle audio input
    if audio_input is not None:
        # Save audio file
        uploads_dir = Path(LOGDIR) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        src_path = getattr(audio_input, "name", audio_input) if hasattr(audio_input, "name") else audio_input
        if src_path and os.path.exists(src_path):
            dst_path = uploads_dir / f"audio_{int(time.time())}_{random.randint(1000, 9999)}.wav"
            shutil.copy2(src_path, dst_path)
            state.audio_files.append(str(dst_path))
            state.add_message("user", (str(dst_path),))
        else:
            state.add_message("user", text if text else "[Audio]")
    else:
        state.add_message("user", text)
    
    state.add_message("assistant", None)
    state.skip_next = False
    
    # Enable buttons after adding user message
    return (state, state.to_chatbot_format(), "", None) + (enable_btn,) * 2

def add_uploaded_audio(state, file, use_emotion, request: gr.Request):
    """
    处理 UploadButton 上传的音/视频文件。
    直接复用 add_text 的逻辑，把上传的 file 当作 audio_input 传进去，
    text 设为 ""，表示这是纯语音消息。
    """
    return add_text(state, "", file, use_emotion, request)

def http_bot(
    state, 
    model_selector, 
    temperature, 
    top_p, 
    max_new_tokens,
    use_emotion,
    enable_tts,
    tts_role,
    tts_speed,
    tts_split_mode,
    controller_url: str
):
    """Generate response via HTTP to model worker with streaming."""
    logger.info(f"http_bot. model={model_selector}")
    start_time = time.time()
    
    if state is None or not isinstance(state, ConversationState):
        state = ConversationState()
    
    if not state.messages:
        state.add_message("assistant", "会话状态已重置，请重新发送消息。")
        yield (state, state.to_chatbot_format()) + (enable_btn,) * 2 + (None,)
        return
    
    if state.skip_next:
        yield (state, state.to_chatbot_format()) + (no_change_btn,) * 2 + (None,)
        return
    
    model_name = model_selector
    
    # Get worker address
    worker_addr = get_worker_address(controller_url, model_name)
    if not worker_addr:
        state.messages[-1][-1] = SERVER_ERROR_MSG
        yield (state, state.to_chatbot_format()) + (enable_btn,) * 2 + (None,)
        return
    
    logger.info(f"Using worker: {worker_addr}")
    
    # Build request payload
    last_user_msg = None
    audio_path = None
    for role, content in reversed(state.messages[:-1]):
        if role == "user":
            if isinstance(content, tuple):
                audio_path = content[0]
                last_user_msg = ""
            else:
                last_user_msg = content
            break
    
    pload = {
        "model": model_name,
        "prompt": last_user_msg or "",
        "audio_path": audio_path,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "use_emotion": use_emotion,
    }
    logger.info(f"Request: model={model_name}, has_audio={audio_path is not None}")
    
    # Update TTS settings
    tts_manager.enabled = enable_tts
    tts_manager.role = tts_role
    tts_manager.speed = tts_speed
    tts_manager.split_mode = tts_split_mode or "punctuation"
    
    # Setup TTS streaming
    tts_text_queue = queue.Queue() if enable_tts else None
    if enable_tts and tts_text_queue:
        tts_manager.start_streaming_tts(tts_text_queue)
    
    # Show loading indicator
    if state.messages[-1][0] != "assistant":
        state.add_message("assistant", None)
    state.messages[-1][-1] = "▌"
    yield (state, state.to_chatbot_format()) + (disable_btn,) * 2 + (None,)
    
    # Stream response with true audio streaming
    generated_text = ""
    prev_text = ""
    
    # Track audio streaming state
    audio_stream_buffer = []  # Buffer for collecting audio files
    last_yielded_audio_idx = -1  # Track which audio was last sent to UI
    full_audio_path = None  # Final concatenated audio
    
    # Timing control for sequential playback
    # 增加额外缓冲时间，确保浏览器端音频完整播放（考虑网络传输+浏览器加载延迟）
    INTER_SEGMENT_PAUSE = 0.15  # Natural pause between segments (seconds)
    BROWSER_LOAD_BUFFER = 0.5   # 浏览器加载和开始播放的预估延迟
    next_playback_time = time.time()  # When we can start next audio
    
    def flush_audio_queue():
        """Yield any buffered audio segments in order, respecting playback gaps."""
        nonlocal last_yielded_audio_idx, next_playback_time
        while last_yielded_audio_idx + 1 < len(audio_stream_buffer):
            next_idx = last_yielded_audio_idx + 1
            audio_path = audio_stream_buffer[next_idx]
            
            wait_time = next_playback_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            
            duration = tts_manager.get_audio_duration(audio_path)
            # 加入浏览器加载缓冲时间，确保前一段音频在浏览器端完整播放
            next_playback_time = time.time() + duration + INTER_SEGMENT_PAUSE + BROWSER_LOAD_BUFFER
            last_yielded_audio_idx = next_idx
            
            yield (state, state.to_chatbot_format()) + (disable_btn,) * 2 + (
                gr.update(value=audio_path, autoplay=True),
            )
    
    try:
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=120
        )
        
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                try:
                    data = json.loads(chunk.decode())
                    if data.get("error_code", 0) == 0:
                        generated_text = data.get("text", "")
                        state.messages[-1][-1] = generated_text + "▌"
                        
                        # Send new text to TTS
                        if enable_tts and tts_text_queue:
                            new_text = generated_text[len(prev_text):]
                            if new_text:
                                tts_text_queue.put(new_text)
                            prev_text = generated_text
                        
                        # True streaming: yield new audio with timing control
                        if enable_tts:
                            # Check for new audio chunks
                            new_audio_received = False
                            while True:
                                audio = tts_manager.get_audio(timeout=0.01)
                                if audio == "empty":
                                    break
                                elif audio is None:
                                    # End of TTS stream
                                    break
                                else:
                                    # New audio chunk available
                                    audio_stream_buffer.append(audio)
                                    new_audio_received = True
                            
                            # If we have new audio, yield it with sequential playback
                            if new_audio_received:
                                for update in flush_audio_queue():
                                    yield update
                            else:
                                # No new audio yet, just update text
                                yield (state, state.to_chatbot_format()) + (disable_btn,) * 2 + (gr.update(),)
                        else:
                            # TTS disabled, just stream text
                            yield (state, state.to_chatbot_format()) + (disable_btn,) * 2 + (None,)
                    else:
                        error_msg = data.get("text", SERVER_ERROR_MSG)
                        state.messages[-1][-1] = f"{error_msg} (error: {data.get('error_code')})"
                        yield (state, state.to_chatbot_format()) + (enable_btn,) * 2 + (None,)
                        return
                except json.JSONDecodeError:
                    continue
                
                time.sleep(0.02)
                
    except Exception as e:
        logger.error(f"Request failed: {e}")
        state.messages[-1][-1] = SERVER_ERROR_MSG
        yield (state, state.to_chatbot_format()) + (enable_btn,) * 2 + (None,)
        return
    
    # Signal end of text to TTS
    if enable_tts and tts_text_queue:
        tts_text_queue.put(None)
    
    # Collect remaining audio chunks and yield them with timing control
    if enable_tts:
        # 增加超时机制，防止TTS线程挂起导致无限等待
        remaining_wait_start = time.time()
        MAX_REMAINING_WAIT = 60  # 最多等待60秒
        while time.time() - remaining_wait_start < MAX_REMAINING_WAIT:
            audio = tts_manager.get_audio(timeout=1.0)
            if audio == "empty":
                if tts_manager.tts_thread and tts_manager.tts_thread.is_alive():
                    continue
                break
            elif audio is None:
                break
            else:
                audio_stream_buffer.append(audio)
                state.messages[-1][-1] = generated_text + "▌"
                for update in flush_audio_queue():
                    yield update
        
        # Ensure any buffered audio is flushed
        state.messages[-1][-1] = generated_text + "▌"
        for update in flush_audio_queue():
            yield update
        
        # Wait for last audio segment to finish before finalizing
        wait_time = next_playback_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Concatenate full audio for replay
        full_audio_path = tts_manager.concat_audio_files(audio_stream_buffer)
    
    # Finalize response
    state.messages[-1][-1] = generated_text
    final_audio_update = None
    if enable_tts:
        final_audio_value = full_audio_path or (audio_stream_buffer[-1] if audio_stream_buffer else None)
        final_audio_update = gr.update(value=final_audio_value, autoplay=False)
    
    # Log conversation
    end_time = time.time()
    logger.info(f"Response generated in {end_time - start_time:.2f}s")
    
    try:
        with open(get_conv_log_filename(), "a") as f:
            log_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": model_name,
                "duration": round(end_time - start_time, 2),
                "has_audio_input": audio_path is not None,
                "use_emotion": use_emotion,
                "enable_tts": enable_tts,
            }
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log conversation: {e}")
    
    # Final yield - enable buttons for next interaction
    yield (state, state.to_chatbot_format()) + (enable_btn,) * 2 + (final_audio_update,)


def regenerate(state, request: gr.Request):
    """Regenerate last response."""
    logger.info(f"regenerate. ip: {request.client.host}")
    if state is None or not isinstance(state, ConversationState):
        state = ConversationState()
        state.add_message("assistant", "会话状态已重置，请重新开始对话。")
        return (state, state.to_chatbot_format(), "") + (enable_btn,) * 2 + (None,)
    if state.messages and len(state.messages) >= 2:
        state.messages[-1][-1] = None
    state.skip_next = False
    # Keep buttons enabled during regeneration
    return (state, state.to_chatbot_format(), "") + (enable_btn,) * 2 + (None,)


# ============================================================
# Gradio UI
# ============================================================

title_markdown = """
<div style="display: flex; align-items: center; padding: 20px; border-radius: 10px; background-color: #f0f4f8;">
  <div style="margin-right: 20px;">
    <h1 style="margin: 0; color: #1a73e8;">🎭 Emo-Omni Chat</h1>
    <h3 style="margin: 10px 0 0 0; color: #5f6368;">Empathic Speech Understanding & Generation</h3>
  </div>
</div>
"""

tos_markdown = """
## Terms of Use
This is a research preview for non-commercial use only. 
The service may collect dialogue data for research purposes.
"""

css = """
#chatbot {
    height: 500px;
    overflow-y: auto;
}
.message-row img {
    margin: 0px !important;
}
"""


def build_demo(controller_url: str, concurrency_count: int = 10):
    """Build the Gradio demo interface."""
    
    models = get_model_list(controller_url)
    
    with gr.Blocks(title="Emo-Omni Chat", theme=gr.themes.Soft(), css=css) as demo:
        state = gr.State(ConversationState())
        
        gr.Markdown(title_markdown)
        
        with gr.Row():
            # Left column - settings
            with gr.Column(scale=3):
                model_selector = gr.Dropdown(
                    choices=models,
                    value=models[0] if models else "",
                    label="Model",
                    interactive=True
                )
                
                with gr.Accordion("⚙️ Generation Settings", open=True):
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=DEFAULT_TEMPERATURE,
                        step=0.1, label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=DEFAULT_TOP_P,
                        step=0.05, label="Top P"
                    )
                    max_new_tokens = gr.Slider(
                        minimum=64, maximum=2048, value=DEFAULT_MAX_NEW_TOKENS,
                        step=64, label="Max New Tokens"
                    )
                
                with gr.Accordion("🎭 Emotion & Voice Settings", open=True):
                    use_emotion = gr.Checkbox(
                        label="🎭 Use Emotion-Aware Mode",
                        value=True,
                        info="Enable empathic responses based on user's emotion"
                    )
                    enable_tts = gr.Checkbox(
                        label="🔊 Enable TTS Voice Reply",
                        value=True,
                        info="Generate voice response (requires TTS server)"
                    )
                    tts_role = gr.Dropdown(
                        choices=TTS_VOICE_LIST,
                        value=DEFAULT_TTS_ROLE,
                        label="Voice Role"
                    )
                    tts_speed = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0,
                        step=0.1, label="TTS Speed"
                    )
                    tts_split_mode = gr.Radio(
                        choices=[
                            ("按句号/问号等标点切分（推荐）", "punctuation"),
                            ("快速模式（逗号等也切分）", "fast"),
                        ],
                        value="punctuation",
                        label="TTS 文本切分模式"
                    )
                
                refresh_btn = gr.Button("🔄 Refresh Models")
            
            # Right column - chat
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Chat",
                    height=450,
                    show_copy_button=True
                )
                
                # TTS audio output
                tts_audio = gr.Audio(
                    label="🔊 Voice Reply",
                    type="filepath",
                    autoplay=True,
                    visible=True
                )
                
                with gr.Row():
                    textbox = gr.Textbox(
                        show_label=False,
                        placeholder="Type a message or upload audio...",
                        container=False,
                        scale=6,
                    )
                    audio_input = gr.Audio(
                        label="🎤",
                        sources=["microphone", "upload"],
                        type="filepath",
                        scale=2,
                        visible=True,
                    )
                    # ✅ 文件上传按钮（音频/视频）
                    audio_upload = gr.UploadButton(
                        "📁 Upload audio / video",
                        file_types=["audio", "video"],
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("💬 Send", variant="primary")
                    regenerate_btn = gr.Button("🔄 Regenerate", interactive=False)
                    clear_btn = gr.Button("🗑️ Clear", interactive=False)
        
        gr.Markdown(tos_markdown)
        
        # Event handlers
        btn_list = [regenerate_btn, clear_btn]
        
        def refresh_models():
            models = get_model_list(controller_url)
            return gr.Dropdown(choices=models, value=models[0] if models else "")
        
        refresh_btn.click(refresh_models, [], [model_selector])
        
        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, audio_input] + btn_list + [tts_audio],
            queue=False
        )
        
        # Submit handlers
        submit_inputs = [state, textbox, audio_input, use_emotion]
        submit_outputs = [state, chatbot, textbox, audio_input] + btn_list
        
        bot_inputs = [
            state, model_selector, temperature, top_p, max_new_tokens,
            use_emotion, enable_tts, tts_role, tts_speed, tts_split_mode
        ]
        bot_outputs = [state, chatbot] + btn_list + [tts_audio]
        
        # ============================================================
        # 包装函数：解决 lambda 返回 generator 的问题
        # ============================================================
        def bot_response_wrapper(*args):
            """
            包装函数：使用 'yield from' 逐个传递 http_bot 生成的值。
            """
            yield from http_bot(*args, controller_url=controller_url)

        textbox.submit(
            add_text,
            submit_inputs,
            submit_outputs,
            queue=False
        ).then(
            bot_response_wrapper,
            bot_inputs,
            bot_outputs,
            queue=True
        )
        
        submit_btn.click(
            add_text,
            submit_inputs,
            submit_outputs,
            queue=False
        ).then(
            bot_response_wrapper,
            bot_inputs,
            bot_outputs,
            queue=True
        )

        audio_upload.upload(
            add_uploaded_audio,
            [state, audio_upload, use_emotion],
            submit_outputs,
            queue=False,
        ).then(
            bot_response_wrapper,
            bot_inputs,
            bot_outputs,
            queue=True,
        )
        
        # 重新生成
        regenerate_btn.click(
            regenerate,
            [state],
            [state, chatbot, textbox] + btn_list + [tts_audio]
        ).then(
            bot_response_wrapper,
            bot_inputs,
            bot_outputs,
            queue=True
        )
        
        # Load models on startup
        demo.load(
            refresh_models,
            [],
            [model_selector],
            queue=False
        )
    
    return demo



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emomni Gradio Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_WEBUI_PORT)
    parser.add_argument(
        "--controller-url", 
        type=str, 
        default=f"http://localhost:{DEFAULT_CONTROLLER_PORT}"
    )
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"Starting Gradio server on {args.host}:{args.port}")
    logger.info(f"Controller URL: {args.controller_url}")
    
    warmup_tts_service(tts_manager)
    
    demo = build_demo(args.controller_url, args.concurrency_count)
    demo.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=["/"]
    )
