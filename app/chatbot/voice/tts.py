import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import pyttsx3

logger = logging.getLogger(__name__)


class BaseTextToSpeech(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        pass

    @abstractmethod
    def save_to_file(self, text: str, filename: str) -> bool:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class TerminalTextToSpeech(BaseTextToSpeech):
    def __init__(
        self, voice_id: Optional[str] = None, rate: int = 180, volume: float = 1.0
    ):
        self.engine = pyttsx3.init()

        if voice_id:
            self.engine.setProperty("voice", voice_id)

        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def is_available(self) -> bool:
        try:
            voices = self.engine.getProperty("voices")
            return len(voices) > 0
        except Exception as e:
            logger.error(f"TTS engine not available: {e}")
            return False

    def speak(self, text: str) -> None:
        try:
            logger.info(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error during TTS conversion: {e}")

    def save_to_file(self, text: str, filename: str) -> bool:
        try:
            logger.info(f"Saving speech to file: {filename}")
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            return os.path.exists(filename)
        except Exception as e:
            logger.error(f"Error saving speech to file: {e}")
            return False

    def get_available_voices(self) -> List[Tuple[str, str]]:
        voices = self.engine.getProperty("voices")
        return [(voice.id, voice.name) for voice in voices]


def get_text_to_speech(env: str = "terminal", **kwargs) -> BaseTextToSpeech:
    if env == "terminal":
        return TerminalTextToSpeech(**kwargs)
    if env == "web":
        return TerminalTextToSpeech(**kwargs)
    else:
        raise ValueError(f"Unsupported environment: {env}")
