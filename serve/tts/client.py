"""Sentence scheduling and streaming speech client for the WebUI."""
import logging
import os
import queue
import re
import threading
import uuid
from pathlib import Path
from typing import List, Tuple
import requests
import numpy as np
import soundfile as sf
from ..constants import LOGDIR, DEFAULT_TTS_API_URL, DEFAULT_TTS_ROLE, TTS_MODES, KOKORO_DEFAULT_VOICE, KOKORO_VOICES
logger = logging.getLogger(__name__)

class TTSManager:
    """TTS Manager for streaming voice reply generation via HTTP API."""

    PUNCT_ALL = r'[。！？；：.!?;:]'
    FAST_SPLIT_ALL = r'[，。！？；：,.!?;:]'

    def __init__(
        self,
        enabled: bool = True,
        role: str = DEFAULT_TTS_ROLE,
        split_mode: str = "punctuation",
        speed: float = 1.0,
        api_url: str = DEFAULT_TTS_API_URL,
        tts_mode: str = TTS_MODES[0],
        clone_ref_audio: str = None,
        clone_ref_text: str = "",
    ):
        self.enabled = enabled
        self.role = KOKORO_DEFAULT_VOICE if tts_mode == "Kokoro" and role not in KOKORO_VOICES else role
        self.split_mode = split_mode
        self.speed = speed
        self.api_url = api_url.rstrip("/")

        self.tts_mode = tts_mode
        # 音色克隆相关（仅 CosyVoice3 模式使用）
        self.clone_ref_audio = clone_ref_audio
        self.clone_ref_text = clone_ref_text

        self.client = requests.Session()
        if os.getenv("TTS_API_KEY"):
            self.client.headers["Authorization"] = f"Bearer {os.environ['TTS_API_KEY']}"

        self.audio_queue = queue.Queue(maxsize=256)
        self.tts_thread = None
        self.stop_signal = False
        self.response = None
        self.finished = threading.Event()
        self.errors = []

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

        remaining_text = remaining[start:] if start < len(remaining) else ""
        return sentences, remaining_text

    def _save_audio(self, data, sample_rate=None):
        path = self.output_dir / f"tts-{uuid.uuid4().hex}.wav"
        if sample_rate is None:
            path.write_bytes(data)
            if sf.info(path).frames == 0:
                raise ValueError("TTS returned empty audio")
        else:
            sf.write(path, np.frombuffer(data, dtype="<i2"), sample_rate, subtype="PCM_16")
        if getattr(self, "asset_callback", None):
            self.asset_callback(path)
        return str(path)

    def synthesize(self, text, timeout=60):
        """Yield playable WAV chunks as PCM arrives; retain WAV for other speeds."""
        if not self.enabled or not text.strip():
            return
        streaming = self.tts_mode == "CosyVoice3" and self.speed == 1
        fmt = "pcm" if streaming else "wav"
        payload = {"input": text, "voice": self.role if self.tts_mode == "Kokoro" else "default",
                   "model": self.tts_mode.lower(), "speed": self.speed, "response_format": fmt,
                   "stream": streaming}
        if self.tts_mode == "CosyVoice3" and self.clone_ref_audio:
            if not self.clone_ref_text.strip():
                raise ValueError("Reference text is required for voice cloning")
            with open(self.clone_ref_audio, "rb") as reference:
                response = self.client.post(self.api_url.removesuffix("/v1") + "/clone",
                    data={"text": text, "speed": self.speed, "reference_text": self.clone_ref_text,
                          "response_format": fmt, "stream": str(streaming).lower()},
                    files={"reference_audio": ("reference.wav", reference)},
                    stream=streaming, timeout=timeout)
        else:
            response = self.client.post(self.api_url + "/audio/speech", json=payload,
                                        stream=streaming, timeout=timeout)
        self.response = response
        try:
            with response:
                response.raise_for_status()
                if self.stop_signal:
                    return
                if not streaming:
                    yield self._save_audio(response.content)
                    return
                rate = int(response.headers.get("X-Audio-Sample-Rate", 24000))
                pending = bytearray()
                emitted = False
                for data in response.iter_content(chunk_size=None):
                    if self.stop_signal:
                        return
                    pending.extend(data)
                    # HTTP/proxy fragments need not end at PCM sample boundaries.
                    while len(pending) >= rate // 2:
                        # Bound playback segments for a one-second HLS target.
                        size = min(len(pending) // 2 * 2, int(rate * 0.9) * 2)
                        yield self._save_audio(pending[:size], rate)
                        del pending[:size]
                        emitted = True
                if len(pending) % 2:
                    raise ValueError("Truncated PCM audio")
                if pending:
                    yield self._save_audio(pending, rate)
                elif not emitted:
                    raise ValueError("TTS returned empty audio")
        finally:
            self.response = None

    def start_streaming_tts(self, text_queue: queue.Queue):
        """Start streaming TTS in background thread."""
        if not self.enabled:
            return

        self.stop_signal = False
        self.finished.clear()
        self.audio_queue = queue.Queue(maxsize=256)

        def _tts_worker():
            buffer = ""
            try:
                while not self.stop_signal:
                    try:
                        text_chunk = text_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if text_chunk is None:
                        sentences, buffer = ([buffer] if buffer.strip() else []), ""
                    else:
                        buffer += text_chunk
                        sentences, buffer = self.split_sentences(buffer, self.split_mode)
                        # Bound unpunctuated input without losing whitespace between streamed chunks.
                        while len(buffer) > 240:
                            split_at = buffer.rfind(" ", 0, 240)
                            split_at = split_at if split_at > 80 else 240
                            sentences.append(buffer[:split_at])
                            buffer = buffer[split_at:]
                    for sentence in sentences:
                        if self.stop_signal:
                            break
                        if not any(character.isalnum() for character in sentence):
                            continue
                        for audio_path in self.synthesize(sentence):
                            if self.stop_signal:
                                Path(audio_path).unlink(missing_ok=True)
                                continue
                            try:
                                self.audio_queue.put_nowait(audio_path)
                            except queue.Full:
                                self.errors.append("tts_backlog")
                                self.stop_streaming_tts()
                                Path(audio_path).unlink(missing_ok=True)
                    if text_chunk is None:
                        break
            except Exception as exc:
                if not self.stop_signal:
                    self.errors.append(str(exc))
                    logger.exception("TTS worker failed")
            finally:
                self.finished.set()
                try:
                    self.audio_queue.put_nowait(None)
                except queue.Full:
                    pass

        self.tts_thread = threading.Thread(target=_tts_worker)
        self.tts_thread.daemon = True
        self.tts_thread.start()

    def stop_streaming_tts(self):
        """Stop streaming TTS."""
        self.stop_signal = True
        response = self.response
        if response is not None:
            # Closing a socket may wait for an in-flight read; keep the UI responsive.
            threading.Thread(target=response.close, daemon=True).start()

    def get_audio(self, timeout: float = 0.1):
        """Get next audio file from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return "empty"

    def concat_audio_files(self, audio_files):
        """Build replay audio without re-encoding or dropping PCM boundaries."""
        if not audio_files:
            return None
        if len(audio_files) == 1:
            return audio_files[0]
        path = self.output_dir / f"tts-concat-{uuid.uuid4().hex}.wav"
        try:
            info = sf.info(audio_files[0])
            with sf.SoundFile(path, "w", samplerate=info.samplerate,
                              channels=info.channels, subtype="PCM_16") as output:
                for source in audio_files:
                    with sf.SoundFile(source) as audio:
                        if audio.samplerate != info.samplerate or audio.channels != info.channels:
                            raise ValueError("Inconsistent TTS audio format")
                        for block in audio.blocks(blocksize=65536, dtype="int16"):
                            output.write(block)
            return str(path)
        except Exception:
            path.unlink(missing_ok=True)
            logger.exception("Audio concatenation failed")
            return None
