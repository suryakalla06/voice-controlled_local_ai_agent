from pathlib import Path

from src.voice_agent.config import AppConfig
from src.voice_agent.llm import LLMService
from src.voice_agent.models import AgentResult, CommandStep, IntentResult
from src.voice_agent.stt import STTService
from src.voice_agent.tools import ToolExecutor


class VoiceAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.stt = STTService(config)
        self.llm = LLMService(config)
        self.tools = ToolExecutor(config.output_dir)

    def prepare(self, audio_path: Path) -> AgentResult:
        transcript = self.stt.transcribe(audio_path)
        intent = self.llm.classify(transcript)
        requires_confirmation = any(
            step.primary_intent in {"create_file", "write_code", "write_text"}
            for step in intent.steps
        )
        action_summary = self._describe_action(intent)
        output = intent.response_text or f"Ready to execute {len(intent.steps)} step(s)."
        return AgentResult(
            transcript=transcript,
            intent=intent,
            action_summary=action_summary,
            output=output,
            requires_confirmation=requires_confirmation,
        )

    def execute(self, prepared_result: AgentResult) -> AgentResult:
        outputs: list[str] = []
        created_paths: list[Path] = []
        last_created_folder: Path | None = None
        last_text_result: str | None = None

        for step in prepared_result.intent.steps:
            effective_step = self._apply_context(step, last_created_folder, last_text_result)
            output, paths = self.tools.execute(effective_step)
            outputs.append(output)
            created_paths.extend(paths)

            if effective_step.primary_intent == "create_file" and effective_step.create_folder and paths:
                last_created_folder = paths[0]
            if effective_step.primary_intent in {"summarize_text", "general_chat", "unsupported_intent"}:
                last_text_result = output
            if effective_step.primary_intent == "write_text" and effective_step.content:
                last_text_result = effective_step.content

        prepared_result.output = "\n".join(outputs) if outputs else prepared_result.output
        prepared_result.created_paths = created_paths
        prepared_result.requires_confirmation = False
        return prepared_result

    def _apply_context(
        self,
        step: CommandStep,
        last_created_folder: Path | None,
        last_text_result: str | None,
    ) -> CommandStep:
        updates: dict[str, str] = {}

        if last_created_folder is not None and step.target_path:
            raw_target = step.target_path.replace("\\", "/")
            if "/" not in raw_target and step.primary_intent in {"create_file", "write_code", "write_text"}:
                relative_folder = last_created_folder.resolve().relative_to(self.config.output_dir.resolve())
                updates["target_path"] = str(relative_folder / raw_target)

        if step.primary_intent == "write_text" and (step.content is None or not step.content.strip()) and last_text_result:
            updates["content"] = last_text_result

        return step.model_copy(update=updates) if updates else step

    @staticmethod
    def _describe_action(intent: IntentResult) -> str:
        lines = [VoiceAgent._describe_step(step) for step in intent.steps]
        return "\n".join(lines)

    @staticmethod
    def _describe_step(step: CommandStep) -> str:
        if step.primary_intent == "create_file":
            if step.create_folder:
                return f"Create folder `{step.target_path or 'generated_folder'}` in the output directory."
            return f"Create file `{step.target_path or 'generated.txt'}` in the output directory."
        if step.primary_intent == "write_code":
            language = step.language.upper()
            return f"Generate {language} code and save it to `{step.target_path or 'generated.txt'}`."
        if step.primary_intent == "write_text":
            return f"Write text to `{step.target_path or 'generated.txt'}` in the output directory."
        if step.primary_intent == "summarize_text":
            return "Summarize the requested content."
        if step.primary_intent == "unsupported_intent":
            return "Detect and report an unsupported request safely."
        return "Respond as a general chat assistant."
