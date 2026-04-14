from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


IntentType = Literal[
    "create_file",
    "write_code",
    "write_text",
    "summarize_text",
    "general_chat",
    "unsupported_intent",
]


class CommandStep(BaseModel):
    primary_intent: IntentType = "general_chat"
    target_path: str | None = None
    content: str | None = None
    summary_source: str | None = None
    response_text: str | None = None
    language: str = "python"
    create_folder: bool = False


class IntentResult(BaseModel):
    intents: list[str] = Field(default_factory=list)
    primary_intent: IntentType = "general_chat"
    target_path: str | None = None
    content: str | None = None
    summary_source: str | None = None
    response_text: str | None = None
    language: str = "python"
    create_folder: bool = False
    steps: list[CommandStep] = Field(default_factory=list)


class AgentResult(BaseModel):
    transcript: str
    intent: IntentResult
    action_summary: str
    output: str
    requires_confirmation: bool = False
    created_paths: list[Path] = Field(default_factory=list)
