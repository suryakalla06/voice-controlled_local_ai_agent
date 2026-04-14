import unittest

from src.voice_agent.config import AppConfig
from src.voice_agent.llm import LLMService


class LLMNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LLMService(
            AppConfig(
                output_dir=None,
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

    def test_normalize_payload_coerces_list_and_null_values(self) -> None:
        payload = {
            "content": ["line 1", "line 2"],
            "response_text": None,
            "summary_source": ["alpha", "beta"],
            "language": None,
            "create_folder": "true",
            "steps": [
                {"primary_intent": "create_file", "target_path": "new_f", "create_folder": True},
                {"primary_intent": "write_text", "target_path": "new_f/notes.txt", "content": ["hello", "world"]},
            ],
        }

        normalized = self.service._normalize_payload(payload)

        self.assertEqual(normalized["content"], "line 1\nline 2")
        self.assertIsNone(normalized["response_text"])
        self.assertEqual(normalized["summary_source"], "alpha\nbeta")
        self.assertEqual(normalized["language"], "text")
        self.assertTrue(normalized["create_folder"])
        self.assertEqual(len(normalized["steps"]), 2)
        self.assertEqual(normalized["steps"][1]["content"], "hello\nworld")

    def test_unknown_intent_maps_to_unsupported(self) -> None:
        self.assertEqual(self.service._coerce_intent("delete_system_files"), "unsupported_intent")

    def test_coerce_language_maps_aliases(self) -> None:
        self.assertEqual(self.service._coerce_language("c++"), "cpp")
        self.assertEqual(self.service._coerce_language(["ts"]), "typescript")
        self.assertEqual(self.service._coerce_language("plain_text"), "text")

    def test_infer_language_from_target_path(self) -> None:
        self.assertEqual(self.service._infer_language("src/main.cpp"), "cpp")
        self.assertEqual(self.service._infer_language("retry.py"), "python")
        self.assertIsNone(self.service._infer_language(None))

    def test_strip_code_fences(self) -> None:
        content = "```cpp\nint main() { return 0; }\n```"
        self.assertEqual(self.service._strip_code_fences(content), "int main() { return 0; }")


if __name__ == "__main__":
    unittest.main()
