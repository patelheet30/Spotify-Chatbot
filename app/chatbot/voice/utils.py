import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_temp_audio_file(prefix: str = "speech_", extension: str = "mp3") -> str:
    directory = os.path.join(tempfile.gettempdir(), "spotify_chatbot_audio")
    os.makedirs(directory, exist_ok=True)

    filename = f"{prefix}{uuid.uuid4().hex}.{extension}"
    return os.path.join(directory, filename)


def play_audio_file(filename: str) -> bool:
    if not os.path.exists(filename):
        logger.error(f"Audio file not found: {filename}")
        return False
    try:
        import platform

        system = platform.system()

        if system == "Windows":
            import subprocess

            subprocess.Popen(["start", filename], shell=True)
        elif system == "Darwin":
            import subprocess

            subprocess.Popen(["afplay", filename])
        else:
            logger.warning(f"Unsupported OS: {system}. Cannot play audio.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error playing audio file: {e}")
        return False
