# Emomni Serve Constants

import os
from pathlib import Path

# Log directory
LOGDIR = os.environ.get("EMOMNI_LOGDIR", str(Path(__file__).parent.parent / "logs" / "serve"))

# Controller settings
CONTROLLER_HEART_BEAT_EXPIRATION = 30  # seconds

# Worker settings
WORKER_HEART_BEAT_INTERVAL = 15  # seconds

# Default ports
DEFAULT_CONTROLLER_PORT = 21001
DEFAULT_WORKER_PORT = 21002
DEFAULT_WEBUI_PORT = 7860

# TTS settings
DEFAULT_TTS_API_URL = os.environ.get("TTS_API_URL", "http://127.0.0.1:8880/v1")
DEFAULT_TTS_ROLE = KOKORO_DEFAULT_VOICE = "zf_xiaoxiao"
TTS_MODES = ["Kokoro", "CosyVoice3"]
KOKORO_VOICES = {'zf_xiaobei': ('晓北 · 中文女声', 'Xiaobei · Chinese female'),
 'zf_xiaoxiao': ('晓晓 · 中文女声', 'Xiaoxiao · Chinese female'),
 'zm_yunjian': ('云健 · 中文男声', 'Yunjian · Chinese male'),
 'zm_yunxi': ('云希 · 中文男声', 'Yunxi · Chinese male'),
 'af_heart': ('Heart · 美式女声', 'Heart · American female'),
 'am_adam': ('Adam · 美式男声', 'Adam · American male'),
 'bf_emma': ('Emma · 英式女声', 'Emma · British female')}

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
SERVER_ERROR_MSG = '**NETWORK ERROR: The server encountered an issue. Please retry.**'
MODERATION_MSG = '**INPUT MODERATION: Your input was flagged. Please adjust and try again.**'
