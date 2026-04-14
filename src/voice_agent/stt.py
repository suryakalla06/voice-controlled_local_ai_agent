from pathlib import Path

from groq import Groq
from openai import OpenAI

from src.voice_agent.config import AppConfig


class STTService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def transcribe(self, audio_path: Path) -> str:
        if self.config.stt_provider == "groq":
            if not self.config.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when STT_PROVIDER=groq")
            client = Groq(api_key=self.config.groq_api_key)
            with audio_path.open("rb") as file_handle:
                transcription = client.audio.transcriptions.create(
                    file=(audio_path.name, file_handle.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
            return transcription.text

        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when STT_PROVIDER=openai")
        client = OpenAI(api_key=self.config.openai_api_key)
        with audio_path.open("rb") as file_handle:
            transcription = client.audio.transcriptions.create(
                file=file_handle,
                model="gpt-4o-mini-transcribe",
            )
        return transcription.text
