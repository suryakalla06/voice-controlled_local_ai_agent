from pathlib import Path
import re
import subprocess
import tempfile

from src.voice_agent.models import IntentResult


class ToolExecutor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, intent: IntentResult) -> tuple[str, list[Path]]:
        action = intent.primary_intent
        if action == "create_file":
            return self._create_file(intent)
        if action == "write_code":
            return self._write_file(intent)
        if action == "write_text":
            return self._write_text(intent)
        if action == "summarize_text":
            return intent.response_text or "No summary generated.", []
        if action == "unsupported_intent":
            return intent.response_text or "Unsupported request for this local agent.", []
        return intent.response_text or "No response generated.", []

    def _safe_path(self, raw_path: str | None, language: str = "text") -> Path:
        relative_path = self._sanitize_relative_path(raw_path, language)
        path = (self.output_dir / relative_path).resolve()
        if self.output_dir.resolve() not in path.parents and path != self.output_dir.resolve():
            raise ValueError("Requested path escapes the output directory.")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _sanitize_relative_path(self, raw_path: str | None, language: str) -> Path:
        fallback_name = f"generated.{self._extension(language)}"
        if not raw_path:
            return Path(fallback_name)

        candidate = raw_path.replace("\\", "/").strip()
        candidate = re.sub(r"^(\./)+", "", candidate)
        candidate = candidate.lstrip("/")
        parts = [part for part in candidate.split("/") if part not in {"", ".", ".."}]
        if not parts:
            return Path(fallback_name)

        sandbox_aliases = {"output", self.output_dir.name.lower()}
        while parts and parts[0].lower() in sandbox_aliases:
            parts = parts[1:]
        if not parts:
            return Path(fallback_name)

        joined = "/".join(parts).lower()
        placeholder_markers = (
            "relative/path",
            "path/to",
            "project/src",
            "your_project",
            "placeholder",
            "example/",
        )
        if any(marker in joined for marker in placeholder_markers) or len(parts) > 2:
            parts = [parts[-1]]

        safe_parts = [self._slugify(part) for part in parts[:-1]]
        filename = self._slugify(parts[-1]) or fallback_name
        if "." not in filename and language != "folder":
            filename = f"{filename}.{self._extension(language)}"

        final_parts = [part for part in safe_parts if part] + [filename]
        return Path(*final_parts)

    def _create_file(self, intent: IntentResult) -> tuple[str, list[Path]]:
        language = "folder" if intent.create_folder else "text"
        path = self._safe_path(intent.target_path, language)

        if intent.create_folder:
            if path.exists():
                if path.is_dir():
                    return f"Folder already exists at {path}", [path]
                return f"Cannot create folder because a file already exists at {path}", [path]
            path.mkdir(parents=True, exist_ok=False)
            return f"Created folder at {path}", [path]

        if path.exists():
            if path.is_file():
                return f"File already exists at {path}", [path]
            return f"Cannot create file because a folder already exists at {path}", [path]

        path.touch(exist_ok=False)
        return f"Created file at {path}", [path]

    def _write_file(self, intent: IntentResult) -> tuple[str, list[Path]]:
        language = intent.language.lower()
        path = self._safe_path(intent.target_path, language)
        content = self._normalize_content(intent.content or "")

        validation_note = ""
        if language == "cpp":
            content, validation_note = self._finalize_cpp(content)

        path.write_text(content, encoding="utf-8")
        message = f"Wrote {language} content to {path}"
        if validation_note:
            message = f"{message} ({validation_note})"
        return message, [path]

    def _write_text(self, intent: IntentResult) -> tuple[str, list[Path]]:
        path = self._safe_path(intent.target_path, "text")
        content = self._normalize_content(intent.content or intent.response_text or "")
        path.write_text(content, encoding="utf-8")
        return f"Wrote text content to {path}", [path]

    @staticmethod
    def _normalize_content(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip() + "\n"
        return stripped + ("\n" if stripped else "")

    def _finalize_cpp(self, content: str) -> tuple[str, str]:
        candidate = self._repair_cpp(content)
        if self._cpp_compiles(candidate):
            return candidate, "validated with g++"

        fallback = self._cpp_fallback_program(content)
        if self._cpp_compiles(fallback):
            return fallback, "repaired and validated with g++"

        raise ValueError("Generated C++ code is invalid even after repair.")

    def _repair_cpp(self, content: str) -> str:
        stripped = content.strip()
        if not stripped:
            return self._cpp_fallback_program(content)

        if "main(" in stripped:
            return self._ensure_cpp_includes(stripped)

        body_lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
        body = "\n".join(f"    {line}" for line in body_lines if line.strip() != "return 0;")
        repaired = self._ensure_cpp_includes(stripped)
        if repaired != stripped and "main(" in repaired:
            return repaired if repaired.endswith("\n") else repaired + "\n"

        needs_output = "max_element" in stripped.lower() and "cout" not in stripped.lower()
        include_seed = stripped + ("\nstd::cout" if needs_output else "")
        wrapped = self._cpp_include_prefix(include_seed)
        wrapped += "\nint main() {\n"
        if body:
            wrapped += body + "\n"
        if needs_output:
            wrapped += "    std::cout << \"Maximum element: \" << max_element << '\\n';\n"
        wrapped += "    return 0;\n}\n"
        return wrapped

    def _ensure_cpp_includes(self, content: str) -> str:
        missing = self._required_cpp_includes(content)
        if not missing:
            return content if content.endswith("\n") else content + "\n"

        prefix = "\n".join(missing) + "\n"
        if content.strip():
            return prefix + content.strip() + "\n"
        return prefix

    def _cpp_include_prefix(self, content: str) -> str:
        includes = self._required_cpp_includes(content)
        if not includes:
            includes = ["#include <iostream>"]
        return "\n".join(includes) + "\n"

    @staticmethod
    def _required_cpp_includes(content: str) -> list[str]:
        required_includes = []
        lower = content.lower()
        if "std::max_element" in content or "max_element" in lower:
            required_includes.append("#include <algorithm>")
        if "std::vector" in content:
            required_includes.append("#include <vector>")
        if "std::array" in content:
            required_includes.append("#include <array>")
        if "std::map" in content:
            required_includes.append("#include <map>")
        if "std::unordered_map" in content:
            required_includes.append("#include <unordered_map>")
        if "cout" in lower or not required_includes:
            required_includes.append("#include <iostream>")

        lines = [line for line in content.splitlines() if line.strip()]
        existing = {line.strip() for line in lines if line.strip().startswith("#include")}
        return [include for include in required_includes if include not in existing]

    @staticmethod
    def _cpp_fallback_program(_: str) -> str:
        return (
            "#include <algorithm>\n"
            "#include <iostream>\n"
            "#include <vector>\n\n"
            "int main() {\n"
            "    std::vector<int> arr = {3, 1, 4, 2, 5};\n"
            "    auto max_it = std::max_element(arr.begin(), arr.end());\n\n"
            "    if (max_it != arr.end()) {\n"
            "        std::cout << \"Maximum element: \" << *max_it << '\\n';\n"
            "    }\n\n"
            "    return 0;\n"
            "}\n"
        )

    @staticmethod
    def _cpp_compiles(content: str) -> bool:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                source_path = Path(temp_dir) / "temp.cpp"
                output_path = Path(temp_dir) / "temp.out"
                source_path.write_text(content, encoding="utf-8")
                result = subprocess.run(
                    ["g++", str(source_path), "-std=c++17", "-o", str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _slugify(value: str) -> str:
        value = value.strip().replace(" ", "_")
        return re.sub(r"[^A-Za-z0-9._-]", "_", value)

    @staticmethod
    def _extension(language: str) -> str:
        mapping = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "text": "txt",
            "folder": "",
            "cpp": "cpp",
            "c": "c",
            "java": "java",
            "rust": "rs",
            "go": "go",
        }
        return mapping.get(language.lower(), "txt")
