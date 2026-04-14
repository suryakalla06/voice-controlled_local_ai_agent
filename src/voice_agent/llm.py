import json
from pathlib import Path

import requests
from groq import Groq
from openai import OpenAI

from src.voice_agent.config import AppConfig
from src.voice_agent.models import CommandStep, IntentResult


INTENT_PROMPT = """You are an intent router for a local AI voice agent.
Return only valid JSON with this schema:
{
  "intents": ["create_file" | "write_code" | "write_text" | "summarize_text" | "general_chat" | "unsupported_intent"],
  "primary_intent": "create_file" | "write_code" | "write_text" | "summarize_text" | "general_chat" | "unsupported_intent",
  "target_path": "relative/path/inside/output/or null",
  "content": "content to write if needed",
  "summary_source": "text to summarize if present",
  "response_text": "assistant response for chat or summaries",
  "language": "python",
  "create_folder": false,
  "steps": [
    {
      "primary_intent": "create_file" | "write_code" | "write_text" | "summarize_text" | "general_chat" | "unsupported_intent",
      "target_path": "relative/path/inside/output/or null",
      "content": "content to write if needed",
      "summary_source": "text to summarize if present",
      "response_text": "assistant response for chat or summaries",
      "language": "python",
      "create_folder": false
    }
  ]
}

Rules:
- Keep file paths relative, never absolute.
- For a single request, you may use either the top-level fields or steps.
- For compound commands, always use steps in execution order. Support as many steps as needed.
- Use write_code only for source code.
- Use write_text for plain text content, including saving summaries, notes, or chat output to a file.
- For write_code and write_text, return only raw file contents in content. Do not use markdown fences.
- For write_code, generate complete, runnable, syntactically valid code for the requested language.
- Infer language from the requested filename when possible, for example .py -> python, .cpp -> cpp, .js -> javascript, .ts -> typescript.
- If the user asks to create an empty file, use create_file and set create_folder to false.
- If the user asks to create a folder or directory, use create_file and set create_folder to true.
- When a later step says to save something inside a folder from an earlier step, prefer giving the later step the full relative path.
- If the user asks for a summary, use summarize_text and put the actual summary in response_text.
- If a later step saves a summary or chat response to a file, use write_text with content equal to the text to save.
- If the request is conversational, use general_chat and put the answer in response_text.
- If the request asks for a capability this app does not support safely, use unsupported_intent and explain the limitation in response_text.
- When unsure, prefer unsupported_intent over inventing risky actions.
"""


class LLMService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def classify(self, transcript: str) -> IntentResult:
        if self.config.llm_provider == "ollama":
            payload = {
                "model": self.config.ollama_model,
                "prompt": f"{INTENT_PROMPT}\n\nUser request:\n{transcript}",
                "stream": False,
                "format": "json",
            }
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["response"]
        elif self.config.llm_provider == "groq":
            if not self.config.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
            client = Groq(api_key=self.config.groq_api_key)
            completion = client.chat.completions.create(
                model=self.config.groq_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": transcript},
                ],
            )
            content = completion.choices[0].message.content
        else:
            if not self.config.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            client = OpenAI(api_key=self.config.openai_api_key)
            completion = client.responses.create(
                model=self.config.openai_model,
                temperature=0,
                text={"format": {"type": "json_object"}},
                input=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": transcript},
                ],
            )
            content = completion.output_text

        raw = json.loads(content)
        normalized = self._normalize_payload(raw)
        parsed = IntentResult.model_validate(normalized)
        return self._normalize_intent(parsed)

    def _normalize_payload(self, payload: dict) -> dict:
        normalized = dict(payload)
        normalized["content"] = self._coerce_text(normalized.get("content"))
        normalized["response_text"] = self._coerce_text(normalized.get("response_text"))
        normalized["summary_source"] = self._coerce_text(normalized.get("summary_source"))
        normalized["language"] = self._coerce_language(normalized.get("language"))
        normalized["create_folder"] = self._coerce_bool(normalized.get("create_folder"))
        normalized["primary_intent"] = self._coerce_intent(normalized.get("primary_intent"))
        steps = normalized.get("steps") or []
        normalized["steps"] = [self._normalize_step_payload(step) for step in steps if isinstance(step, dict)]
        return normalized

    def _normalize_step_payload(self, payload: dict) -> dict:
        normalized = dict(payload)
        normalized["content"] = self._coerce_text(normalized.get("content"))
        normalized["response_text"] = self._coerce_text(normalized.get("response_text"))
        normalized["summary_source"] = self._coerce_text(normalized.get("summary_source"))
        normalized["language"] = self._coerce_language(normalized.get("language"))
        normalized["create_folder"] = self._coerce_bool(normalized.get("create_folder"))
        normalized["primary_intent"] = self._coerce_intent(normalized.get("primary_intent"))
        return normalized

    def _normalize_intent(self, intent: IntentResult) -> IntentResult:
        normalized_steps: list[CommandStep] = []
        if intent.steps:
            for step in intent.steps:
                normalized_steps.append(self._normalize_step(step))
        else:
            normalized_steps.append(
                self._normalize_step(
                    CommandStep(
                        primary_intent=intent.primary_intent,
                        target_path=intent.target_path,
                        content=intent.content,
                        summary_source=intent.summary_source,
                        response_text=intent.response_text,
                        language=intent.language,
                        create_folder=intent.create_folder,
                    )
                )
            )

        intent.steps = normalized_steps
        intent.intents = list(dict.fromkeys(step.primary_intent for step in normalized_steps))
        intent.primary_intent = normalized_steps[0].primary_intent if normalized_steps else intent.primary_intent

        first = normalized_steps[0] if normalized_steps else None
        if first is not None:
            intent.target_path = first.target_path
            intent.content = first.content
            intent.summary_source = first.summary_source
            intent.response_text = first.response_text
            intent.language = first.language
            intent.create_folder = first.create_folder

        return intent

    def _normalize_step(self, step: CommandStep) -> CommandStep:
        inferred_language = self._infer_language(step.target_path)
        if inferred_language and step.primary_intent == "write_code":
            step.language = inferred_language

        if step.primary_intent in {"write_code", "write_text"} and step.content:
            step.content = self._strip_code_fences(step.content)
            step.create_folder = False

        if step.primary_intent == "unsupported_intent":
            step.target_path = None
            step.create_folder = False

        return step

    @staticmethod
    def _coerce_text(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return str(value)

    @staticmethod
    def _coerce_language(value: object) -> str:
        if value is None:
            return "text"
        if isinstance(value, list):
            value = value[0] if value else "text"
        value = str(value).strip().lower()
        if not value or value == "none":
            return "text"
        aliases = {
            "c++": "cpp",
            "cplusplus": "cpp",
            "js": "javascript",
            "ts": "typescript",
            "plain_text": "text",
        }
        return aliases.get(value, value)

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "folder", "directory"}
        return bool(value)

    @staticmethod
    def _coerce_intent(value: object) -> str:
        allowed = {
            "create_file",
            "write_code",
            "write_text",
            "summarize_text",
            "general_chat",
            "unsupported_intent",
        }
        if value is None:
            return "unsupported_intent"
        value = str(value).strip().lower()
        return value if value in allowed else "unsupported_intent"

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _infer_language(target_path: str | None) -> str | None:
        if not target_path:
            return None

        suffix = Path(target_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".java": "java",
            ".rs": "rust",
            ".go": "go",
            ".txt": "text",
            ".md": "text",
        }
        return mapping.get(suffix)
