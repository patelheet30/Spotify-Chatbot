import logging
from abc import ABC, abstractmethod
from typing import Optional

import speech_recognition as sr
from speech_recognition.recognizers.google import recognize_legacy

logger = logging.getLogger(__name__)


class BaseSpeechRecogniser(ABC):
    @abstractmethod
    def listen(self) -> Optional[str]:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class TerminalSpeechRecogniser(BaseSpeechRecogniser):
    def __init__(
        self,
        language: str = "en-GB",
        timeout: int = 5,
        phrase_time_limit: Optional[int] = None,
    ):
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.recogniser = sr.Recognizer()

    def is_available(self) -> bool:
        try:
            microphone = sr.Microphone.list_microphone_names()
            return len(microphone) > 0
        except (sr.RequestError, OSError) as e:
            logger.error(f"Microphone not available: {e}")
            return False

    def listen(self) -> Optional[str]:
        try:
            with sr.Microphone() as microphone:
                logger.info("Handling Background Noise...")
                self.recogniser.adjust_for_ambient_noise(microphone, duration=1)

                logger.info("Listening...")
                audio = self.recogniser.listen(
                    microphone,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

                logger.info("Recognising...")
                text = recognize_legacy(
                    recognizer=self.recogniser,
                    audio_data=audio,  # type: ignore
                    language=self.language,
                )
                logger.info(f"Recognised text: {text}")
                return text  # type: ignore
        except sr.WaitTimeoutError:
            logger.error("Listening timed out")
            return None
        except sr.UnknownValueError:
            logger.error("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(
                f"Could not request results from Google Speech Recognition service: {e}"
            )
            return None
        except OSError as e:
            logger.error(f"Microphone not available: {e}")
            return None


def get_speech_recogniser(env: str = "terminal", **kwargs) -> BaseSpeechRecogniser:
    if env == "terminal":
        return TerminalSpeechRecogniser(**kwargs)
    elif env == "web":
        return TerminalSpeechRecogniser(**kwargs)
    else:
        raise ValueError(f"Unsupported environment: {env}")
