from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass
class AppConfig:
    output_dir: Path
    stt_provider: str
    llm_provider: str
    openai_api_key: str | None
    groq_api_key: str | None
    ollama_base_url: str
    ollama_model: str
    openai_model: str
    groq_model: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            output_dir=Path(os.getenv("OUTPUT_DIR", "output")).resolve(),
            stt_provider=os.getenv("STT_PROVIDER", "openai"),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        )
