"""
Emomni Gradio Web Server
Provides a web UI for interacting with Emomni models.
Features streaming generation and TTS voice reply.
"""

import argparse
import json
import math
from functools import wraps
import os
import time
import uuid
import queue
import threading
import shutil
import copy
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import gradio as gr
import requests

from ..constants import (
    LOGDIR,
    DEFAULT_WEBUI_PORT,
    DEFAULT_CONTROLLER_PORT,
    DEFAULT_TTS_API_URL,
    TTS_MODES,
    KOKORO_DEFAULT_VOICE,
    KOKORO_VOICES,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from .i18n import I18N
from ..utils import build_logger, violates_moderation
from ..tts.client import TTSManager

logger = build_logger("gradio_web_server", "gradio_web_server.log")

# HTTP headers
headers = {"User-Agent": "Emomni Client"}

# Button state helpers
no_change_btn = gr.update()
enable_btn = gr.update(interactive=True)
disable_btn = gr.update(interactive=False)


# Each request creates an independent TTS manager.
tts_manager = TTSManager()


class ConversationState:
    """Manages conversation state for the web UI."""

    def __init__(self):
        self.session_id = uuid.uuid4().hex
        self.busy = False
        self.active_run_id = None
        self.turn_id = ""
        self.failed_assistant_indices = set()
        self.messages = []  # List of (role, content) tuples
        self.audio_files = []  # List of audio file paths
        self.current_tts_segments = []  # 当前回答已生成的 TTS 分段
        self.last_tts_audio = None  # 最近一次可回放音频（分段或拼接后）
        self.skip_next = False

    def reset(self):
        """Reset conversation state."""
        self.messages = []
        self.audio_files = []
        self.current_tts_segments = []
        self.last_tts_audio = None
        self.skip_next = False
        self.busy = False
        self.turn_id = ""
        self.failed_assistant_indices = set()

    def add_message(self, role: str, content):
        """Add a message to the conversation."""
        self.messages.append([role, content])

    def add_user_message(self, text: str = "", audio_path: Optional[str] = None):
        """Add a user message that may contain both text and audio."""
        payload = {
            "text": text or "",
            "audio_path": audio_path or "",
        }
        self.messages.append(["user", payload])

    def _normalize_user_content(self, content) -> Dict[str, str]:
        """Normalize legacy and structured user message payloads."""
        if isinstance(content, dict):
            return {
                "text": content.get("text", "") or "",
                "audio_path": content.get("audio_path", "") or "",
            }
        if isinstance(content, tuple):
            audio_path = content[0] if len(content) > 0 else ""
            text = content[1] if len(content) > 1 and isinstance(content[1], str) else ""
            return {
                "text": text or "",
                "audio_path": audio_path or "",
            }
        return {
            "text": content or "",
            "audio_path": "",
        }

    def serialize_messages(self, include_pending_assistant: bool = False) -> List[Dict[str, Any]]:
        """Serialize conversation messages for the model worker."""
        serialized = []
        for index, (role, content) in enumerate(self.messages):
            if role == "user":
                if index + 1 in self.failed_assistant_indices:
                    continue
                payload = self._normalize_user_content(content)
                if payload["text"] or payload["audio_path"]:
                    serialized.append({
                        "role": "user",
                        "text": payload["text"],
                        "audio_path": payload["audio_path"],
                    })
            elif role == "assistant":
                if index in self.failed_assistant_indices:
                    continue
                if content is None and not include_pending_assistant:
                    continue
                serialized.append({
                    "role": "assistant",
                    "text": content or "",
                })
        return serialized

    def to_chatbot_format(self) -> List:
        """Convert messages to Gradio chatbot format."""
        chatbot = []
        for role, content in self.messages:
            if role == "user":
                payload = self._normalize_user_content(content)
                if payload["audio_path"] and payload["text"]:
                    # Keep the audio player visible while still showing the paired text prompt.
                    chatbot.append([(payload["audio_path"],), None])
                    display_content = payload["text"]
                elif payload["audio_path"]:
                    display_content = (payload["audio_path"],)
                else:
                    display_content = payload["text"]
                chatbot.append([display_content, None])
            elif role == "assistant":
                if chatbot and chatbot[-1][1] is None:
                    chatbot[-1][1] = content
                else:
                    chatbot.append([None, content])
        return chatbot

    def copy(self):
        """Create a copy of the conversation state."""
        new_state = ConversationState()
        new_state.messages = copy.deepcopy(self.messages)
        new_state.audio_files = list(self.audio_files)
        new_state.current_tts_segments = list(self.current_tts_segments)
        new_state.last_tts_audio = self.last_tts_audio
        new_state.skip_next = self.skip_next
        new_state.failed_assistant_indices = set(self.failed_assistant_indices)
        return new_state


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

# Live threads and HTTP clients cannot be stored in deepcopy-based gr.State.
# Each state receives its own random session id via the State callable factory.
_runs = {}
_runs_lock = threading.Lock()
_session_assets = {}
_session_assets_lock = threading.Lock()
MAX_INPUT_CHARS = 12000
MAX_AUDIO_BYTES = 20 * 1024 * 1024


def _audio_directories():
    return (Path(LOGDIR) / "tts_output",
            Path(os.getenv("EMOMNI_UPLOAD_DIR", str(Path(tempfile.gettempdir()) / "emomni-uploads"))))


def _track_asset(state, path):
    if path:
        with _session_assets_lock:
            _session_assets.setdefault(state.session_id, set()).add(str(Path(path).resolve()))


def cleanup_expired_audio(ttl_seconds=None):
    """Delete expired original audio only when no live session references it."""
    ttl = max(60, int(os.getenv("EMOMNI_AUDIO_TTL_SECONDS", "86400"))) if ttl_seconds is None else ttl_seconds
    cutoff = time.time() - ttl
    with _session_assets_lock:
        protected = set().union(*_session_assets.values()) if _session_assets else set()
    removed = 0
    for directory in _audio_directories():
        if not directory.is_dir():
            continue
        root = directory.resolve()
        for path in directory.iterdir():
            try:
                resolved = path.resolve()
                if resolved.parent != root or str(resolved) in protected:
                    continue
                if path.is_file() and path.stat().st_mtime < cutoff:
                    with _session_assets_lock:
                        if any(str(resolved) in assets for assets in _session_assets.values()):
                            continue
                        path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                logger.debug("Audio retention cleanup skipped %s", path)
    return removed


def start_audio_cleanup():
    def clean():
        while True:
            try:
                cleanup_expired_audio()
            except Exception:
                logger.exception("Audio retention cleanup failed")
            time.sleep(600)
    threading.Thread(target=clean, daemon=True, name="audio-retention").start()


def dispose_session(state):
    _cancel_run(state)
    if isinstance(state, ConversationState):
        with _session_assets_lock:
            _session_assets.pop(state.session_id, None)


def _translations(lang):
    return I18N.get(lang, I18N["en"])


def _buttons(running=False, has_history=True):
    return (
        gr.update(visible=not running, interactive=not running),
        gr.update(visible=running, interactive=running),
        gr.update(interactive=has_history and not running),
        gr.update(interactive=has_history and not running),
    )


def _cancel_run(state, expected_id=None):
    if not isinstance(state, ConversationState):
        return
    with _runs_lock:
        run = _runs.get(state.session_id)
        if expected_id is not None and (run is None or run["id"] != expected_id):
            return
        _runs.pop(state.session_id, None)
    state.busy = False
    state.active_run_id = None
    state.turn_id = ""
    if not run:
        return
    run["cancelled"].set()
    run["manager"].stop_streaming_tts()
    # Cancellation must reach the model even if the HTTP stream is stalled.
    def cancel_backend():
        try:
            requests.post(run["worker"] + "/worker_cancel", json={"request_id": run["id"]}, timeout=3)
        except requests.RequestException:
            logger.warning("Worker cancellation unavailable")
        response = run.get("response")
        if response is not None:
            response.close()
    threading.Thread(target=cancel_backend, daemon=True).start()


def clear_history(state):
    """Only cancel and clear the requesting browser session."""
    dispose_session(state)
    return (ConversationState(), [], "", None) + _buttons(has_history=False) + (None, None)


def _save_audio(audio_input):
    source = getattr(audio_input, "name", audio_input)
    if not isinstance(source, (str, os.PathLike)):
        raise ValueError("invalid_audio")
    source = Path(source)
    if not source.is_file() or not source.stat().st_size:
        raise ValueError("invalid_audio")
    if source.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("audio_too_large")
    import soundfile as sf
    try:
        info = sf.info(str(source))
    except (RuntimeError, OSError) as exc:
        raise ValueError("invalid_audio") from exc
    if info.duration > 30:
        raise ValueError("audio_too_long")
    # Shared temp directory is accepted by the model worker and container mounts.
    uploads = Path(os.getenv("EMOMNI_UPLOAD_DIR", str(Path(tempfile.gettempdir()) / "emomni-uploads")))
    uploads.mkdir(parents=True, exist_ok=True)
    destination = uploads / f"{uuid.uuid4().hex}{source.suffix.lower() or '.wav'}"
    shutil.copy2(source, destination)
    return str(destination)


def add_text(state, text, audio_input, use_emotion, ui_lang="en"):
    state = state if isinstance(state, ConversationState) else ConversationState()
    t = _translations(ui_lang)
    text = (text or "").strip()
    if state.busy:
        # Raise instead of toggling shared skip_next: a duplicate submit must not
        # suppress or mutate the already queued generation.
        raise gr.Error(t["busy"])
    error = None
    if not text and not audio_input:
        error = "empty_input"
    elif len(text) > MAX_INPUT_CHARS:
        error = "input_too_long"
    elif violates_moderation(text):
        error = "moderation"
    audio_path = None
    if not error and audio_input:
        try:
            audio_path = _save_audio(audio_input)
        except (ValueError, OSError) as exc:
            error = str(exc) if str(exc) in t else "invalid_audio"
    if error:
        state.skip_next = True
        gr.Warning(t[error])
        return (state, state.to_chatbot_format(), text, audio_input) + _buttons(has_history=bool(state.messages))
    state.add_user_message(text, audio_path)
    if audio_path:
        state.audio_files.append(audio_path)
        _track_asset(state, audio_path)
    state.add_message("assistant", None)
    state.skip_next = False
    state.turn_id = uuid.uuid4().hex
    state.busy = True
    state.current_tts_segments = []
    state.last_tts_audio = None
    return (state, state.to_chatbot_format(), "", None) + _buttons(running=True)


def add_uploaded_audio(state, text, file, use_emotion, ui_lang="en"):
    return add_text(state, text, file, use_emotion, ui_lang)


def regenerate(state, ui_lang="en"):
    state = state if isinstance(state, ConversationState) else ConversationState()
    if state.busy:
        raise gr.Error(_translations(ui_lang)["busy"])
    valid = len(state.messages) >= 2 and state.messages[-1][0] == "assistant" and state.messages[-2][0] == "user"
    state.skip_next = not valid
    if valid:
        state.messages[-1][1] = None
        state.failed_assistant_indices.discard(len(state.messages) - 1)
        state.turn_id = uuid.uuid4().hex
        state.busy = True
        state.current_tts_segments = []
        state.last_tts_audio = None
    return (state, state.to_chatbot_format(), "") + _buttons(running=valid, has_history=bool(state.messages)) + (None, None)


def edit_target(state, row, column):
    if not isinstance(state, ConversationState) or state.busy or column != 0:
        return None
    current = 0
    for index, (role, content) in enumerate(state.messages):
        if role != "user":
            continue
        payload = state._normalize_user_content(content)
        count = 2 if payload["audio_path"] and payload["text"] else 1
        if current <= row < current + count:
            return {"session": state.session_id, "index": index, "original": copy.deepcopy(content)}
        current += count
    return None

def edit_history(state, target, text, ui_lang="zh"):
    if not isinstance(state, ConversationState) or state.busy:
        raise gr.Error(_translations(ui_lang)["busy"])
    index = target.get("index", -1) if isinstance(target, dict) else -1
    if (not isinstance(index, int) or index < 0 or index >= len(state.messages)
            or target.get("session") != state.session_id or state.messages[index][0] != "user"
            or target.get("original") != state.messages[index][1]):
        raise gr.Error("Message changed. Select it again." if ui_lang == "en" else "消息已变化，请重新选择。")
    text = (text or "").strip()
    payload = state._normalize_user_content(state.messages[index][1])
    if not text and not payload["audio_path"]:
        raise gr.Error(_translations(ui_lang)["empty_input"])
    if len(text) > MAX_INPUT_CHARS:
        raise gr.Error(_translations(ui_lang)["input_too_long"])
    if violates_moderation(text):
        raise gr.Error(_translations(ui_lang)["moderation"])
    if payload["audio_path"] and not Path(payload["audio_path"]).is_file():
        raise gr.Error(_translations(ui_lang)["invalid_audio"])
    _cancel_run(state)
    state.messages = state.messages[:index]
    state.failed_assistant_indices = {i for i in state.failed_assistant_indices if i < index}
    state.add_user_message(text, payload["audio_path"])
    state.add_message("assistant", None)
    state.turn_id = uuid.uuid4().hex
    state.busy = True
    state.skip_next = False
    state.current_tts_segments = []
    state.last_tts_audio = None
    return (state, state.to_chatbot_format()) + _buttons(running=True) + (None, None)


def stop_generation(state):
    with _runs_lock:
        run = _runs.get(state.session_id) if isinstance(state, ConversationState) else None
    _cancel_run(state)
    if not isinstance(state, ConversationState):
        state = ConversationState()
    state.skip_next = True
    if state.messages and state.messages[-1][0] == "assistant":
        content = state.messages[-1][1]
        state.messages[-1][1] = content.rstrip("▌") if isinstance(content, str) else ""
    if run and state.current_tts_segments:
        state.last_tts_audio = run["manager"].concat_audio_files(list(state.current_tts_segments))
        _track_asset(state, state.last_tts_audio)
    logger.info("UI stop completed; session=%s, had_active_run=%s", state.session_id, run is not None)
    return (state, state.to_chatbot_format()) + _buttons(has_history=bool(state.messages)) + (
        gr.update(value=None, autoplay=False), gr.update(value=state.last_tts_audio, autoplay=False),
    )


def _request_tts_manager(enabled, role, speed, split_mode, mode, ref_audio, ref_text):
    if mode == "Kokoro" and role not in KOKORO_VOICES:
        role = KOKORO_DEFAULT_VOICE
    return TTSManager(enabled=enabled, role=role, speed=speed,
                      split_mode=split_mode or "punctuation", api_url=tts_manager.api_url,
                      tts_mode=mode, clone_ref_audio=ref_audio, clone_ref_text=ref_text or "")


def _worker_chunks(response, cancelled):
    """Poll text without holding up audio chunks while the model is silent."""
    chunks = queue.Queue(maxsize=128)
    stopped = threading.Event()

    def put(item):
        while not stopped.is_set() and not cancelled.is_set():
            try:
                chunks.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def read():
        try:
            for chunk in response.iter_lines(delimiter=b"\0"):
                if stopped.is_set() or cancelled.is_set():
                    break
                if chunk:
                    put(chunk)
        except Exception as exc:
            put(exc)
        finally:
            put(None)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        while not cancelled.is_set():
            try:
                chunk = chunks.get(timeout=0.04)
            except queue.Empty:
                yield b""
                continue
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
    finally:
        stopped.set()


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, use_emotion,
             enable_tts, tts_role, tts_speed, tts_split_mode, tts_mode,
             tts_clone_ref_audio, tts_clone_ref_text, ui_lang, controller_url, turn_id=None):
    """Stream text and audio independently; always release this session's resources."""
    t = _translations(ui_lang)
    state = state if isinstance(state, ConversationState) else ConversationState()
    if turn_id is not None and turn_id != state.turn_id:
        yield (gr.skip(),) * 8
        return
    if state.skip_next or not state.messages:
        state.busy = False
        yield (state, state.to_chatbot_format()) + _buttons(has_history=bool(state.messages)) + (gr.skip(), gr.skip())
        return
    if state.messages[-1][0] != "assistant":
        state.add_message("assistant", None)
    assistant_index = len(state.messages) - 1
    worker = get_worker_address(controller_url, model_selector) if model_selector else ""
    if not worker:
        state.messages[assistant_index][1] = t["no_model" if not model_selector else "server_error"]
        state.failed_assistant_indices.add(assistant_index)
        state.busy = False
        yield (state, state.to_chatbot_format()) + _buttons() + (None, None)
        return
    manager = _request_tts_manager(enable_tts, tts_role, tts_speed, tts_split_mode,
                                   tts_mode or TTS_MODES[0], tts_clone_ref_audio, tts_clone_ref_text)
    run = {"id": turn_id or uuid.uuid4().hex, "worker": worker, "manager": manager,
           "cancelled": threading.Event(), "response": None}
    with _runs_lock:
        if state.session_id in _runs:
            manager.client.close()
            return
        _runs[state.session_id] = run
    state.active_run_id = run["id"]
    manager.asset_callback = lambda path: _track_asset(state, path) if not run["cancelled"].is_set() else None
    messages = state.serialize_messages()
    latest_user = next((m for m in reversed(messages) if m["role"] == "user"), {})
    payload = {"model": model_selector, "messages": messages, "prompt": latest_user.get("text", ""),
               "audio_path": latest_user.get("audio_path") or None, "temperature": float(temperature),
               "top_p": float(top_p), "max_new_tokens": int(max_new_tokens), "use_emotion": use_emotion,
               "request_id": run["id"]}
    text_queue = queue.Queue(maxsize=4096)
    segments = []
    generated_text = ""
    feedback_code = "stream_error"
    tts_overflow = False
    failed = False
    state.busy = True
    state.messages[assistant_index][1] = "▌"
    state.last_tts_audio = None
    state.current_tts_segments = segments
    started = time.monotonic()

    def update(audio=None, replay=None, done=False):
        return (state, state.to_chatbot_format()) + _buttons(running=not done) + (
            audio if audio is not None else gr.skip(), replay if replay is not None else gr.skip())

    def take_audio():
        while True:
            audio = manager.get_audio(timeout=0)
            if audio is None or audio == "empty":
                break
            segments.append(audio)
            _track_asset(state, audio)
            state.last_tts_audio = audio
            yield audio

    def finish_tts():
        try:
            text_queue.put_nowait(None)
        except queue.Full:
            manager.stop_streaming_tts()

    try:
        yield (state, state.to_chatbot_format()) + _buttons(running=True) + (gr.update(value=None, autoplay=True), None)
        if enable_tts:
            manager.start_streaming_tts(text_queue)
        with requests.post(worker + "/worker_generate_stream", headers=headers, json=payload,
                           stream=True, timeout=(5, 120)) as response:
            run["response"] = response
            try:
                response.raise_for_status()
            except requests.HTTPError:
                try:
                    feedback_code = response.json().get("code", "stream_error")
                except Exception:
                    pass
                raise
            for chunk in _worker_chunks(response, run["cancelled"]):
                if run["cancelled"].is_set():
                    # Gradio replays its cached final diff when a generator returns.
                    # Replace that cache with no-op updates, so it cannot restore
                    # the old running buttons/cursor after stop_generation reset them.
                    yield (gr.skip(),) * 8
                    return
                if not chunk:
                    for audio in take_audio():
                        yield update(audio=audio)
                    continue
                data = json.loads(chunk.decode("utf-8"))
                if data.get("error_code", 0):
                    failed = True
                    # Keep useful partial text; do not send transport diagnostics into future prompts.
                    code = data.get("code", "")
                    message = t.get(code, t["stream_error"])
                    state.messages[assistant_index][1] = generated_text + ("\n\n" if generated_text else "") + message
                    state.failed_assistant_indices.add(assistant_index)
                    break
                new_text = data.get("text", "")
                if not isinstance(new_text, str) or not new_text.startswith(generated_text):
                    raise ValueError("Non-monotonic text stream")
                delta = new_text[len(generated_text):]
                generated_text = new_text
                state.messages[assistant_index][1] = generated_text + "▌"
                if enable_tts and delta and not tts_overflow:
                    try:
                        text_queue.put_nowait(delta)
                    except queue.Full:
                        tts_overflow = True
                        manager.stop_streaming_tts()
                        gr.Warning(t["tts_backlog"])
                yield update()
                for audio in take_audio():
                    yield update(audio=audio)
                if data.get("finish_reason") == "cancelled":
                    break
        if run["cancelled"].is_set():
            yield (gr.skip(),) * 8
            return
        if not failed:
            state.messages[assistant_index][1] = generated_text or t["empty_response"]
            if not generated_text:
                state.failed_assistant_indices.add(assistant_index)
        finish_tts()
        if enable_tts:
            deadline = time.monotonic() + 90
            while not run["cancelled"].is_set():
                for audio in take_audio():
                    yield update(audio=audio)
                if tts_overflow:
                    break
                if manager.finished.is_set() and manager.audio_queue.empty():
                    break
                if time.monotonic() >= deadline:
                    gr.Warning(t["tts_timeout"])
                    manager.stop_streaming_tts()
                    break
                time.sleep(0.04)
            if manager.errors:
                gr.Warning(t["tts_backlog"] if "tts_backlog" in manager.errors else t["tts_failed"])
            if not run["cancelled"].is_set():
                state.last_tts_audio = manager.concat_audio_files(segments)
                _track_asset(state, state.last_tts_audio)
        if run["cancelled"].is_set():
            yield (gr.skip(),) * 8
            return
        state.busy = False
        # The browser queues streaming segments itself. Reply completion never waits
        # for audio playback, and full replay is a separate non-streaming component.
        yield update(replay=gr.update(value=state.last_tts_audio, autoplay=False), done=True)
    except GeneratorExit:
        _cancel_run(state, run["id"])
        raise
    except Exception:
        if not run["cancelled"].is_set():
            logger.exception("Generation request failed")
            state.messages[assistant_index][1] = generated_text + ("\n\n" if generated_text else "") + t.get(feedback_code, t["stream_error"])
            state.failed_assistant_indices.add(assistant_index)
            state.busy = False
            yield update(done=True)
        else:
            yield (gr.skip(),) * 8
    finally:
        finish_tts()
        manager.stop_streaming_tts()
        if run.get("response") is not None:
            run["response"].close()
        # Defer client disposal if one bounded synthesis call is finishing in its thread.
        def close_client():
            if manager.tts_thread:
                manager.tts_thread.join(timeout=65)
            manager.client.close()
        if manager.tts_thread and manager.tts_thread.is_alive():
            threading.Thread(target=close_client, daemon=True).start()
        else:
            manager.client.close()
        with _runs_lock:
            if _runs.get(state.session_id) is run:
                _runs.pop(state.session_id, None)
        if state.active_run_id == run["id"]:
            state.busy = False
            state.active_run_id = None
        logger.info("Generation completed in %.2fs; tts_segments=%d", time.monotonic() - started, len(segments))


css = """
.gradio-container {max-width: 1360px !important; margin:auto; font-family:Inter,ui-sans-serif,system-ui,'Microsoft YaHei',sans-serif !important;}
#hero {padding:10px 8px 8px; border-bottom:1px solid var(--border-color-primary); margin-bottom:8px;}
#hero h1 {font-size:27px; letter-spacing:-1px; margin-bottom:6px;}
#hero p {color:var(--body-text-color-subdued); font-size:15px;}
#chatbot {border-radius:18px;}
#chatbot .message {font-size:15px; line-height:1.7;}
#composer textarea {font-size:16px;}
#settings {gap:10px;}
#connection {font-size:13px; color:var(--body-text-color-subdued);}
#voice_dock {border:1px solid var(--border-color-primary);border-radius:12px;}
#voice_dock .tabitem {padding:4px 8px !important;}
#voice_dock .waveform-container {max-height:52px;}
#edit_hint {font-size:12px;color:var(--body-text-color-subdued);}
#history_editor {padding:12px;border:1px solid var(--border-color-primary);border-radius:12px;}
@media(min-width:900px){#settings {max-height:calc(100vh - 180px);overflow-y:auto;overflow-x:hidden;flex-wrap:nowrap;} #settings > * {flex-shrink:0 !important;} #chatbot {height:clamp(220px,27vh,320px) !important;}}
footer {display:none !important;}
@media(max-width:760px) {#hero {padding:12px 4px;} #hero h1 {font-size:28px;} #chatbot {min-height:340px;}}
"""


def _voice_choices(lang, mode="Kokoro"):
    if mode == "Kokoro":
        return [(labels[0 if lang == "zh" else 1], voice) for voice, labels in KOKORO_VOICES.items()]
    return [("Default", "default")]


def _hero(lang):
    t = _translations(lang)
    return f'# {t["title"]}\n{t["subtitle"]}'


PAUSE_AUDIO_JS = """() => {document.querySelectorAll('#tts_audio_output audio, #tts_replay audio').forEach(a => a.pause());}"""


class SafeStreamingBlocks(gr.Blocks):
    """Correct absent-stream finalization and audio HLS timing in Gradio 5.35."""

    async def handle_streaming_outputs(self, block_fn, data, session_hash, run,
                                       root_path=None, final=False):
        if final and session_hash is not None and run is not None:
            streams = self.pending_streams.get(session_hash, {}).get(run, {})
            data = list(data)
            for index, block in enumerate(block_fn.outputs):
                if isinstance(block, gr.components.StreamingOutput) and block.streaming and block._id not in streams:
                    data[index] = gr.skip()
        result = await super().handle_streaming_outputs(block_fn, data, session_hash, run,
                                                         root_path=root_path, final=final)
        streams = self.pending_streams.get(session_hash, {}).get(run, {})
        for block in block_fn.outputs:
            stream = streams.get(block._id)
            if isinstance(block, gr.Audio) and block.streaming and stream and stream.segments:
                # Gradio increases this with every chunk, delaying playlist polls.
                # PCM chunks are bounded; derive the target from actual durations.
                stream.max_duration = max(1, math.ceil(max(s["duration"] for s in stream.segments)))
        return result


def build_demo(controller_url: str, concurrency_count: int = 10, default_tts_mode: str = None, use_emotion_default=True):
    models = get_model_list(controller_url)
    default_lang = "zh"
    t = _translations(default_lang)
    mode_default = default_tts_mode if default_tts_mode in TTS_MODES else TTS_MODES[0]
    role_default = KOKORO_DEFAULT_VOICE if mode_default == "Kokoro" else "default"
    with SafeStreamingBlocks(title="Emomni · Empathic conversation", theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
                   css=css, delete_cache=(3600, 86400)) as demo:
        state = gr.State(ConversationState, time_to_live=86400, delete_callback=dispose_session)
        ui_lang = gr.State(default_lang)
        turn_token = gr.Textbox(value="", visible=False)
        hero = gr.Markdown(_hero(default_lang), elem_id="hero")
        with gr.Row():
            with gr.Column(scale=3, min_width=260, elem_id="settings"):
                lang_selector = gr.Radio([("中文", "zh"), ("English", "en")], value=default_lang, label=t["lang_label"])
                model_selector = gr.Dropdown(models, value=models[0] if models else None, label=t["model"], interactive=True)
                connection = gr.Markdown(t["connection_ready"] if models else t["connection_empty"], elem_id="connection")
                refresh_btn = gr.Button(t["refresh_models"], size="sm")
                with gr.Accordion(t["emotion_voice"], open=True) as voice_accordion:
                    use_emotion = gr.Checkbox(value=use_emotion_default, label=t["use_emotion"], info=t["use_emotion_info"])
                    enable_tts = gr.Checkbox(value=tts_manager.enabled, label=t["enable_tts"], info=t["enable_tts_info"])
                    tts_mode = gr.State(mode_default)
                    tts_role = gr.Dropdown(_voice_choices(default_lang, mode_default), value=role_default,
                                           label=t["voice_role"], visible=mode_default != "CosyVoice3")
                    with gr.Column(visible=mode_default == "CosyVoice3"):
                        clone_title = gr.Markdown(t["clone_settings_title"])
                        clone_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label=t["clone_ref_audio"])
                        clone_text = gr.Textbox(label=t["clone_ref_text"], placeholder=t["clone_ref_text_placeholder"])
                    tts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label=t["tts_speed"])
                    split_mode = gr.Radio([(t["split_punctuation"], "punctuation"), (t["split_fast"], "fast")],
                                           value="punctuation", label=t["tts_split_mode"])
                with gr.Accordion(t["gen_settings"], open=False) as gen_accordion:
                    temperature = gr.Slider(0, 1.5, value=DEFAULT_TEMPERATURE, step=0.1, label=t["temperature"])
                    top_p = gr.Slider(0.05, 1, value=DEFAULT_TOP_P, step=0.05, label=t["top_p"])
                    max_tokens = gr.Slider(64, 2048, value=DEFAULT_MAX_NEW_TOKENS, step=64, label=t["max_new_tokens"])
            with gr.Column(scale=8, min_width=320):
                chatbot = gr.Chatbot(label=t["chat"], height=360, type="tuples", show_copy_button=True,
                                     elem_id="chatbot", placeholder="Emomni", sanitize_html=True)
                edit_hint = gr.Markdown("点击已发送消息可编辑；保存后移除后续对话并重新生成。原音频保留。", elem_id="edit_hint")
                edit_selection = gr.State(None)
                with gr.Column(visible=False, elem_id="history_editor") as history_editor:
                    edit_text = gr.Textbox(label="编辑消息 / Edit message", lines=2, max_lines=6, interactive=True)
                    with gr.Row():
                        save_edit = gr.Button("保存并重新生成 / Save & regenerate", variant="primary")
                        cancel_edit = gr.Button("取消 / Cancel")
                with gr.Tabs(elem_id="voice_dock"):
                    with gr.Tab("实时播放 / Live"):
                        tts_audio = gr.Audio(label=t["voice_reply"], type="filepath", autoplay=True, streaming=True,
                                             format="wav", elem_id="tts_audio_output")
                    with gr.Tab("回放与下载 / Replay"):
                        replay_audio = gr.Audio(label=t["replay_audio"], type="filepath", autoplay=False,
                                                elem_id="tts_replay")
                playback_hint = gr.Markdown(t["playback_hint"])
                textbox = gr.Textbox(label=t["chat"], show_label=False, lines=2, max_lines=8,
                                     placeholder=t["input_placeholder"], elem_id="composer")
                input_hint = gr.Markdown(t["input_hint"])
                with gr.Accordion("录音与上传 / Audio input", open=False, elem_id="audio_tools"):
                    with gr.Row():
                        audio_input = gr.Audio(label=t["audio_input"], sources=["microphone", "upload"], type="filepath", max_length=30)
                        audio_upload = gr.UploadButton(t["upload_btn"], file_types=["audio"], size="sm", type="filepath")
                with gr.Row():
                    submit_btn = gr.Button(t["send"], variant="primary")
                    stop_btn = gr.Button(t["stop"], variant="stop", visible=False, elem_id="stop_btn")
                    regenerate_btn = gr.Button(t["regenerate"], interactive=False)
                    clear_btn = gr.Button(t["clear"], interactive=False)
        tos = gr.Markdown(t["tos"])

        def refresh_models(lang, current_model):
            available = get_model_list(controller_url)
            chosen = current_model if current_model in available else (available[0] if available else None)
            return gr.update(choices=available, value=chosen), _translations(lang)["connection_ready" if available else "connection_empty"]

        def switch_language(lang, mode, current_model):
            tr = _translations(lang)
            return [lang, _hero(lang), gr.update(label=tr["lang_label"]), gr.update(label=tr["model"]),
                    tr["connection_ready" if current_model else "connection_empty"], gr.update(value=tr["refresh_models"]),
                    gr.update(label=tr["emotion_voice"]), gr.update(label=tr["use_emotion"], info=tr["use_emotion_info"]),
                    gr.update(label=tr["enable_tts"], info=tr["enable_tts_info"]),
                    gr.update(label=tr["voice_role"], choices=_voice_choices(lang, mode)),
                    tr["clone_settings_title"], gr.update(label=tr["clone_ref_audio"]),
                    gr.update(label=tr["clone_ref_text"], placeholder=tr["clone_ref_text_placeholder"]),
                    gr.update(label=tr["tts_speed"]),
                    gr.update(label=tr["tts_split_mode"], choices=[(tr["split_punctuation"], "punctuation"), (tr["split_fast"], "fast")]),
                    gr.update(label=tr["gen_settings"]), gr.update(label=tr["temperature"]), gr.update(label=tr["top_p"]),
                    gr.update(label=tr["max_new_tokens"]), gr.update(label=tr["chat"]),
                    gr.update(placeholder=tr["input_placeholder"], label=tr["chat"]), tr["input_hint"],
                    gr.update(label=tr["audio_input"]), gr.update(label=tr["upload_btn"]),
                    gr.update(value=tr["send"]), gr.update(value=tr["stop"]), gr.update(value=tr["regenerate"]),
                    gr.update(value=tr["clear"]), gr.update(label=tr["voice_reply"]), gr.update(label=tr["replay_audio"]),
                    tr["playback_hint"], tr["tos"]]

        lang_selector.change(switch_language, [lang_selector, tts_mode, model_selector],
                             [ui_lang, hero, lang_selector, model_selector, connection, refresh_btn, voice_accordion,
                              use_emotion, enable_tts, tts_role, clone_title, clone_audio, clone_text,
                              tts_speed, split_mode, gen_accordion, temperature, top_p, max_tokens,
                              chatbot, textbox, input_hint, audio_input, audio_upload, submit_btn, stop_btn,
                              regenerate_btn, clear_btn, tts_audio, replay_audio, playback_hint, tos], queue=False)

        refresh_btn.click(refresh_models, [ui_lang, model_selector], [model_selector, connection])
        demo.load(refresh_models, [ui_lang, model_selector], [model_selector, connection], queue=False)

        buttons = [submit_btn, stop_btn, regenerate_btn, clear_btn]
        def pause_other(selector):
            return "() => {const walk=root=>{if(!root)return;root.querySelectorAll('audio').forEach(a=>a.pause());root.querySelectorAll('*').forEach(e=>{if(e.shadowRoot)walk(e.shadowRoot);});};walk(document.querySelector('" + selector + "'));}"
        replay_audio.play(fn=None, inputs=None, outputs=None, js=pause_other('#tts_audio_output'), queue=False)
        tts_audio.play(fn=None, inputs=None, outputs=None, js=pause_other('#tts_replay'), queue=False)
        def with_turn_token(fn):
            @wraps(fn)
            def prepare(*args):
                result = fn(*args)
                return (*result, result[0].turn_id)
            return prepare

        submit_inputs = [state, textbox, audio_input, use_emotion, ui_lang]
        submit_outputs = [state, chatbot, textbox, audio_input] + buttons + [turn_token]
        bot_inputs = [state, model_selector, temperature, top_p, max_tokens, use_emotion, enable_tts,
                      tts_role, tts_speed, split_mode, tts_mode, clone_audio, clone_text, ui_lang, turn_token]
        bot_outputs = [state, chatbot] + buttons + [tts_audio, replay_audio]
        def bot_wrapper(*args):
            yield from http_bot(*args[:-1], controller_url=controller_url, turn_id=args[-1])
        queue_config = dict(queue=True, concurrency_id="inference", concurrency_limit=concurrency_count, trigger_mode="multiple")
        textbox.submit(with_turn_token(add_text), submit_inputs, submit_outputs, queue=False).success(bot_wrapper, bot_inputs, bot_outputs, **queue_config)
        submit_btn.click(with_turn_token(add_text), submit_inputs, submit_outputs, queue=False).success(bot_wrapper, bot_inputs, bot_outputs, **queue_config)
        # Upload is a draft action; let users preview audio and add text before Send.
        audio_upload.upload(lambda file: file, [audio_upload], [audio_input], queue=False)
        regenerate_btn.click(with_turn_token(regenerate), [state, ui_lang], [state, chatbot, textbox] + buttons + [tts_audio, replay_audio, turn_token],
                                            queue=False).success(bot_wrapper, bot_inputs, bot_outputs, **queue_config)
        def select_for_edit(state, event: gr.SelectData):
            target = edit_target(state, *event.index) if isinstance(event.index, (tuple, list)) and len(event.index) == 2 else None
            if target is None:
                return gr.skip(), gr.skip(), gr.skip()
            return target, state._normalize_user_content(target["original"])["text"], gr.update(visible=True)
        chatbot.select(select_for_edit, [state], [edit_selection, edit_text, history_editor], queue=False)
        cancel_edit.click(lambda: (None, "", gr.update(visible=False)), None, [edit_selection, edit_text, history_editor], queue=False)
        edit_event = save_edit.click(with_turn_token(edit_history), [state, edit_selection, edit_text, ui_lang], bot_outputs + [turn_token], queue=False)
        save_edit.click(fn=None, inputs=None, outputs=None, js=PAUSE_AUDIO_JS, queue=False)
        edit_event.success(lambda: (None, "", gr.update(visible=False)), None, [edit_selection, edit_text, history_editor], queue=False)
        edit_event.success(bot_wrapper, bot_inputs, bot_outputs, **queue_config)
        lang_selector.change(lambda lang: "Click a sent message to edit. Saving removes later turns and regenerates; original audio is retained." if lang == "en" else "点击已发送消息可编辑；保存后移除后续对话并重新生成。原音频保留。", [lang_selector], [edit_hint], queue=False)
        # Cancel by immutable turn token, not Gradio function id: a delayed
        # function-level cancellation can otherwise kill the next submission.
        stop_btn.click(stop_generation, [state], bot_outputs, queue=False)
        stop_btn.click(fn=None, inputs=None, outputs=None, js=PAUSE_AUDIO_JS, queue=False)
        clear_btn.click(clear_history, [state], [state, chatbot, textbox, audio_input] + buttons + [tts_audio, replay_audio],
                        queue=False)
        clear_btn.click(fn=None, inputs=None, outputs=None, js=PAUSE_AUDIO_JS, queue=False)
        # The Chatbot toolbar also exposes Clear; keep server history and audio
        # cancellation consistent with the explicit button below the composer.
        chatbot.clear(clear_history, [state], [state, chatbot, textbox, audio_input] + buttons + [tts_audio, replay_audio],
                      queue=False)
        chatbot.clear(fn=None, inputs=None, outputs=None, js=PAUSE_AUDIO_JS, queue=False)
        for clear_action in [clear_btn.click, chatbot.clear]:
            clear_action(lambda: (None, "", gr.update(visible=False)), None, [edit_selection, edit_text, history_editor], queue=False)
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
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--no-emotion", action="store_true")
    parser.add_argument(
        "--tts-api-url",
        type=str,
        default=DEFAULT_TTS_API_URL,
        help="TTS API base URL, e.g. http://127.0.0.1:8880/v1"
    )
    parser.add_argument(
        "--tts-mode",
        type=str,
        default=TTS_MODES[0],
        choices=TTS_MODES,
        help="Default TTS mode: Kokoro or CosyVoice3"
    )
    args = parser.parse_args()

    # 使用命令行参数配置 TTS Manager
    tts_manager.api_url = args.tts_api_url.rstrip("/")
    tts_manager.tts_mode = args.tts_mode
    tts_manager.enabled = not args.no_tts

    logger.info(f"Starting Gradio server on {args.host}:{args.port}")
    logger.info(f"Controller URL: {args.controller_url}")
    logger.info(f"TTS API URL: {args.tts_api_url}")
    logger.info(f"TTS Mode: {args.tts_mode}")
    start_audio_cleanup()
    demo = build_demo(args.controller_url, args.concurrency_count, default_tts_mode=args.tts_mode, use_emotion_default=not args.no_emotion)
    demo.queue(max_size=20).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str((Path(LOGDIR) / "tts_output").resolve()), os.environ.get("EMOMNI_UPLOAD_DIR", "/tmp/emomni-uploads")],
        max_file_size="20mb"
    )
