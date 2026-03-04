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
DEFAULT_TTS_API_URL = "http://127.0.0.1:8882/v1"
DEFAULT_TTS_ROLE = "中文女"
TTS_VOICE_LIST = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# Error messages
SERVER_ERROR_MSG = "**NETWORK ERROR: The server encountered an issue. Please retry.**"
MODERATION_MSG = "**INPUT MODERATION: Your input was flagged. Please adjust and try again.**"
