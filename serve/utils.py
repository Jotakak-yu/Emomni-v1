# Emomni Serve Utilities

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .constants import LOGDIR

# Ensure log directory exists
Path(LOGDIR).mkdir(parents=True, exist_ok=True)


def build_logger(logger_name: str, logger_filename: str) -> logging.Logger:
    """Build and configure a logger with file and stream handlers.
    
    Args:
        logger_name: Name for the logger instance
        logger_filename: Filename for the log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File handler
    log_file_path = Path(LOGDIR) / logger_filename
    file_handler = logging.FileHandler(str(log_file_path))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def pretty_print_semaphore(semaphore) -> str:
    """Format semaphore state for logging.
    
    Args:
        semaphore: asyncio.Semaphore instance
        
    Returns:
        Formatted string representation
    """
    if semaphore is None:
        return "None"
    
    try:
        value = semaphore._value
        waiters = len(semaphore._waiters) if semaphore._waiters else 0
        return f"value={value}, waiters={waiters}"
    except AttributeError:
        return str(semaphore)


def get_model_display_name(model_path: str) -> str:
    """Extract display name from model path (last two path components).
    
    Args:
        model_path: Full path to the model
        
    Returns:
        Display name (last two path components joined by '/')
    """
    if not model_path:
        return "Unknown"
    
    # Normalize path
    model_path = model_path.rstrip("/")
    parts = model_path.split("/")
    
    # Return last two components
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    elif len(parts) == 1:
        return parts[0]
    else:
        return "Unknown"


def violates_moderation(text: str) -> bool:
    """Check if text violates content moderation policies.
    
    This is a placeholder implementation. In production, this should
    call an actual content moderation API.
    
    Args:
        text: Text to check
        
    Returns:
        True if text violates policies, False otherwise
    """
    # Placeholder - in production, implement actual moderation
    return False
