import tempfile
import unittest
from pathlib import Path

from src.voice_agent.agent import VoiceAgent
from src.voice_agent.config import AppConfig
from src.voice_agent.models import AgentResult, CommandStep, IntentResult
from src.voice_agent.tools import ToolExecutor


class ToolExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.executor = ToolExecutor(self.output_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_strips_duplicate_output_prefix(self) -> None:
        path = self.executor._safe_path("output/map.cpp", "cpp")
        self.assertEqual(path, self.output_dir / "map.cpp")

    def test_flattens_placeholder_paths(self) -> None:
        path = self.executor._safe_path("relative/path/to/project/src/main.cpp", "cpp")
        self.assertEqual(path, self.output_dir / "main.cpp")

    def test_blocks_parent_directory_escape(self) -> None:
        path = self.executor._safe_path("../secret.txt", "text")
        self.assertEqual(path, self.output_dir / "secret.txt")

    def test_adds_extension_when_missing(self) -> None:
        path = self.executor._safe_path("retry", "python")
        self.assertEqual(path, self.output_dir / "retry.py")

    def test_creates_folder_inside_output(self) -> None:
        intent = IntentResult(
            intents=["create_file"],
            primary_intent="create_file",
            target_path="project_docs",
            create_folder=True,
            language="text",
        )
        message, paths = self.executor.execute(intent)
        self.assertIn("Created folder", message)
        self.assertTrue(paths[0].is_dir())
        self.assertEqual(paths[0], self.output_dir / "project_docs")

    def test_create_file_reports_existing_file(self) -> None:
        existing = self.output_dir / "notes.txt"
        existing.write_text("hello", encoding="utf-8")
        intent = IntentResult(
            intents=["create_file"],
            primary_intent="create_file",
            target_path="notes.txt",
            language="text",
        )
        message, paths = self.executor.execute(intent)
        self.assertEqual(message, f"File already exists at {existing}")
        self.assertEqual(paths, [existing])

    def test_write_text_creates_text_file(self) -> None:
        intent = IntentResult(
            intents=["write_text"],
            primary_intent="write_text",
            target_path="summary.txt",
            content="Albert Einstein was a theoretical physicist.",
            language="text",
        )
        message, paths = self.executor.execute(intent)
        self.assertIn("Wrote text content", message)
        self.assertEqual(paths[0].read_text(encoding="utf-8"), "Albert Einstein was a theoretical physicist.\n")

    def test_summary_does_not_create_files(self) -> None:
        intent = IntentResult(
            intents=["summarize_text"],
            primary_intent="summarize_text",
            response_text="Short summary",
            language="text",
        )
        message, paths = self.executor.execute(intent)
        self.assertEqual(message, "Short summary")
        self.assertEqual(paths, [])

    def test_unsupported_intent_returns_safe_message(self) -> None:
        intent = IntentResult(
            intents=["unsupported_intent"],
            primary_intent="unsupported_intent",
            response_text="This request is not supported safely.",
            language="text",
        )
        message, paths = self.executor.execute(intent)
        self.assertEqual(message, "This request is not supported safely.")
        self.assertEqual(paths, [])

    def test_cpp_fragment_is_repaired_and_compiles(self) -> None:
        intent = IntentResult(
            intents=["write_code"],
            primary_intent="write_code",
            target_path="output/max.cpp",
            content="std::array<int, 5> arr = {{3, 1, 4, 2, 5}};\nint max_element = *std::max_element(arr.begin(), arr.end());\nreturn 0;\n",
            language="cpp",
        )

        message, paths = self.executor.execute(intent)
        written = paths[0].read_text(encoding="utf-8")

        self.assertIn("validated with g++", message)
        self.assertEqual(paths[0], self.output_dir / "max.cpp")
        self.assertIn("int main()", written)
        self.assertIn("#include <array>", written)
        self.assertIn("#include <algorithm>", written)

    def test_cpp_map_code_gets_required_includes(self) -> None:
        intent = IntentResult(
            intents=["write_code"],
            primary_intent="write_code",
            target_path="map.cpp",
            content="int main() {\n    std::map<int, int> ordered;\n    std::unordered_map<int, int> unordered;\n    return 0;\n}\n",
            language="cpp",
        )

        _, paths = self.executor.execute(intent)
        written = paths[0].read_text(encoding="utf-8")

        self.assertIn("#include <map>", written)
        self.assertIn("#include <unordered_map>", written)


class CompoundCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.agent = VoiceAgent(
            AppConfig(
                output_dir=self.output_dir,
                stt_provider="openai",
                llm_provider="ollama",
                openai_api_key=None,
                groq_api_key=None,
                ollama_base_url="http://localhost:11434",
                ollama_model="llama3.1:8b",
                openai_model="gpt-4o-mini",
                groq_model="llama-3.1-8b-instant",
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_compound_folder_then_file_uses_folder_context(self) -> None:
        intent = IntentResult(
            intents=["create_file", "write_text"],
            primary_intent="create_file",
            steps=[
                CommandStep(primary_intent="create_file", target_path="new_f", create_folder=True, language="text"),
                CommandStep(primary_intent="write_text", target_path="notes.txt", content="hello", language="text"),
            ],
        )
        prepared = AgentResult(
            transcript="create a folder named new_f and in that new folder create a new file named notes.txt",
            intent=intent,
            action_summary="",
            output="",
            requires_confirmation=True,
        )

        final = self.agent.execute(prepared)
        self.assertTrue((self.output_dir / "new_f").is_dir())
        self.assertTrue((self.output_dir / "new_f" / "notes.txt").is_file())
        self.assertIn(self.output_dir / "new_f" / "notes.txt", final.created_paths)

    def test_summary_then_store_text_file(self) -> None:
        intent = IntentResult(
            intents=["summarize_text", "write_text"],
            primary_intent="summarize_text",
            steps=[
                CommandStep(
                    primary_intent="summarize_text",
                    response_text="Albert Einstein developed the theory of relativity.",
                    language="text",
                ),
                CommandStep(
                    primary_intent="write_text",
                    target_path="alberteinstein.txt",
                    content=None,
                    language="text",
                ),
            ],
        )
        prepared = AgentResult(
            transcript="summarize a paragraph based on Albert Einstein and store the text in a file known as alberteinstein.txt",
            intent=intent,
            action_summary="",
            output="",
            requires_confirmation=True,
        )

        final = self.agent.execute(prepared)
        written = (self.output_dir / "alberteinstein.txt").read_text(encoding="utf-8")
        self.assertIn("Albert Einstein developed the theory of relativity.", written)
        self.assertIn(self.output_dir / "alberteinstein.txt", final.created_paths)

    def test_more_than_two_steps_supported(self) -> None:
        intent = IntentResult(
            intents=["create_file", "write_text", "general_chat"],
            primary_intent="create_file",
            steps=[
                CommandStep(primary_intent="create_file", target_path="session_notes", create_folder=True, language="text"),
                CommandStep(primary_intent="write_text", target_path="todo.txt", content="1. Review transcript", language="text"),
                CommandStep(primary_intent="general_chat", response_text="Done. I created the folder and note file.", language="text"),
            ],
        )
        prepared = AgentResult(
            transcript="create a folder session notes, create a todo file in it, and tell me when done",
            intent=intent,
            action_summary="",
            output="",
            requires_confirmation=True,
        )

        final = self.agent.execute(prepared)
        self.assertTrue((self.output_dir / "session_notes").is_dir())
        self.assertTrue((self.output_dir / "session_notes" / "todo.txt").is_file())
        self.assertIn("Done. I created the folder and note file.", final.output)

    def test_unsupported_intent_is_reported_without_crash(self) -> None:
        intent = IntentResult(
            intents=["unsupported_intent"],
            primary_intent="unsupported_intent",
            steps=[
                CommandStep(
                    primary_intent="unsupported_intent",
                    response_text="This agent cannot send emails or modify system settings.",
                    language="text",
                )
            ],
        )
        prepared = AgentResult(
            transcript="send an email to my team and change my system firewall settings",
            intent=intent,
            action_summary="",
            output="",
            requires_confirmation=False,
        )

        final = self.agent.execute(prepared)
        self.assertEqual(final.output, "This agent cannot send emails or modify system settings.")
        self.assertEqual(final.created_paths, [])


if __name__ == "__main__":
    unittest.main()
