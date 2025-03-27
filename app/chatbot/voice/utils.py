import logging
import os
import platform
import subprocess
import tempfile
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def get_temp_audio_file(
    prefix: str = "speech_", extension: Optional[str] = None
) -> str:
    if extension is None:
        system = platform.system()
        extension = "aiff" if system == "Darwin" else "mp3"

    directory = os.path.join(tempfile.gettempdir(), "spotify_chatbot_audio")
    os.makedirs(directory, exist_ok=True)

    filename = f"{prefix}{uuid.uuid4().hex}.{extension}"
    return os.path.join(directory, filename)


def play_audio_file(filename: str) -> bool:
    if not os.path.exists(filename):
        logger.error(f"Audio file not found: {filename}")
        return False

    try:
        system = platform.system()
        logger.info(f"Playing audio file on {system}: {filename}")

        if system == "Windows":
            subprocess.Popen(["start", filename], shell=True)
            return True

        elif system == "Darwin":
            if filename.endswith(".aiff"):
                subprocess.Popen(["afplay", filename])
                return True
            elif filename.endswith(".mp3"):
                try:
                    subprocess.Popen(["afplay", filename])
                    return True
                except Exception:
                    subprocess.Popen(["open", filename])
                    return True
            else:
                subprocess.Popen(["open", filename])
                return True
        else:
            logger.warning(f"Unsupported OS: {system}. Cannot play audio.")
            return False

    except Exception as e:
        logger.error(f"Error playing audio file: {e}")
        return False
