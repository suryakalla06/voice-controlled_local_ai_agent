# Voice-Controlled Local AI Agent — Engineering Report

A voice-driven agent that transcribes speech, routes it to an intent, and executes
real filesystem actions — with the model's output treated as untrusted at every
step where it could cause damage.

The README covers features and setup. This report covers why the architecture is
shaped the way it is, and where it still falls short.

---

## 1. Problem and scope

An agent that only talks is safe and useless. An agent that acts is useful and
dangerous. Everything difficult about this project sits in that gap.

The specific risk: an LLM asked for a file path will sometimes return
`../../.ssh/authorized_keys`, sometimes `/etc/passwd`, sometimes
`relative/path/to/your_project/main.py` — a placeholder it invented. An LLM asked
for C++ will sometimes return a fragment with no `main()`, or code that
references `std::max_element` without including `<algorithm>`. None of these are
adversarial attacks; they are ordinary model behaviour on an ordinary day.

So the design premise is:

> **Treat every field the model produces as hostile input, and make the damaging
> ones structurally impossible rather than merely unlikely.**

**In scope:** speech-to-text, intent routing across three interchangeable LLM
providers, multi-step compound commands, sandboxed file operations, compile-time
validation of generated C++, and human confirmation before anything is written.

**Out of scope:** running generated code, network access from generated code,
arbitrary shell execution, and persistence beyond the session. These are absent
by choice — each would widen the blast radius past what the safety model covers.

---

## 2. Architecture and data flow

The system is a two-phase pipeline, and the split between the phases is the
whole safety design:

```
  audio (mic or upload)
        │
        ▼
  STTService.transcribe ──► transcript          [groq whisper-large-v3-turbo
        │                                        | openai gpt-4o-mini-transcribe]
        ▼
  LLMService.classify ────► raw JSON            [ollama (local, default)
        │                                        | groq | openai — all JSON mode]
        ▼
  _normalize_payload / _coerce_* ──► IntentResult (Pydantic, closed enum)
        │
        ▼
  ═══════════ VoiceAgent.prepare() ═══════════   ← NOTHING HAS BEEN WRITTEN YET
        │        returns a plan + action_summary
        │        requires_confirmation = True if any step touches the filesystem
        ▼
  ┌─── user reads the plan and confirms ───┐
        │
        ▼
  ═══════════ VoiceAgent.execute() ═══════════
        │
        ▼
  ToolExecutor ─┬─ _safe_path      → path confined to output/, or ValueError
                ├─ _finalize_cpp   → g++ -std=c++17 must succeed, or ValueError
                └─ write
```

`prepare()` and `execute()` are separate methods, not a flag on one method. The
plan is fully computed — including the human-readable `action_summary` describing
each step — before any code path that can write exists on the stack. A confirmation
prompt bolted inside a single `run()` is easy to bypass by accident; two methods
where only the second touches disk is not.

---

## 3. Module walkthrough

| File | What it owns |
|---|---|
| `src/voice_agent/models.py` | `IntentType` (a closed `Literal` of 6), `CommandStep`, `IntentResult`, `AgentResult`. |
| `src/voice_agent/config.py` | `AppConfig.from_env()` — every provider, model and path from environment with defaults. |
| `src/voice_agent/stt.py` | Transcription behind one interface, two providers. |
| `src/voice_agent/llm.py` | The intent prompt, three provider call paths, and the whole normalisation layer. |
| `src/voice_agent/agent.py` | `prepare` / `execute`, and context threading between steps. |
| `src/voice_agent/tools.py` | The sandbox, path sanitisation, file operations, and C++ validation. |
| `app.py` | Streamlit UI — transcript, intent, planned action, confirm button, output. |
| `tests/` | **20 test functions** across `test_tools.py` (15) and `test_llm.py` (5). |

---

## 4. Design decisions

### 4.1 The sandbox is a resolved-path assertion, not a string check

String checks on paths lose. `..` can be encoded, symlinks resolve elsewhere,
Windows separators slip through. `_safe_path` resolves first and then asserts
containment against the resolved sandbox root:

```python
path = (self.output_dir / relative_path).resolve()
if self.output_dir.resolve() not in path.parents and path != self.output_dir.resolve():
    raise ValueError("Requested path escapes the output directory.")
```

Because it compares `.resolve()`d paths, it holds regardless of how the escape
was spelled. `test_blocks_parent_directory_escape` pins it.

### 4.2 Sanitisation before that, for the ordinary failures

`_sanitize_relative_path` runs first and handles the mundane cases the assertion
would reject with an error the user cannot act on:

- backslashes normalised, leading `./` and `/` stripped
- `..`, `.` and empty segments dropped outright
- a leading `output/` alias removed, so the model saying `output/x.py` does not
  produce `output/output/x.py` (`test_strips_duplicate_output_prefix`)
- **invented placeholder paths flattened** — `relative/path`, `path/to`,
  `your_project`, `example/`, or anything deeper than two levels collapses to
  just the filename (`test_flattens_placeholder_paths`)
- every remaining segment slugified to `[A-Za-z0-9._-]`
- an extension inferred from the language when the model omitted one

Two layers with different jobs: sanitisation makes the common case work, the
assertion makes the dangerous case impossible.

### 4.3 Generated C++ is compiled before it is written

This is the part I would most want to be asked about. Generated code that does
not compile is worse than no code, because the user discovers it later and cannot
tell whether the agent or the compiler is at fault. So `_finalize_cpp` refuses to
write C++ it has not proved compiles:

```python
candidate = self._repair_cpp(content)
if self._cpp_compiles(candidate):
    return candidate, "validated with g++"
fallback = self._cpp_fallback_program(content)
if self._cpp_compiles(fallback):
    return fallback, "repaired and validated with g++"
raise ValueError("Generated C++ code is invalid even after repair.")
```

`_cpp_compiles` shells out to `g++ -std=c++17` in a `TemporaryDirectory` with a
15-second timeout and checks the return code. The repair step scans the source
for `std::vector`, `std::map`, `std::unordered_map`, `std::array`,
`max_element`, `cout` and prepends whatever includes are missing; if there is no
`main()`, it wraps the fragment in one.

The ladder is deliberate — **repair, then fall back, then refuse**. It never
writes unvalidated C++, and it never silently writes something unrelated to what
was asked without saying so: the return message carries `validated with g++` or
`repaired and validated with g++` so the user knows which happened.

### 4.4 Three providers behind one interface, local by default

`LLM_PROVIDER` selects Ollama (default, `llama3.1:8b`, local), Groq, or OpenAI;
`STT_PROVIDER` selects Groq or OpenAI. All three LLM paths force structured
output — Ollama's `"format": "json"`, Groq's and OpenAI's `json_object` response
format — so no path relies on the model choosing to emit clean JSON.

Local-by-default matters for what this app does: intent routing sees every
spoken command, and the default configuration keeps that on the machine.

### 4.5 A normalisation layer, because JSON mode is not enough

Structured output guarantees *syntax*, not *sanity*. A model will still return
`"language": ["python"]` instead of a string, `"create_folder": "true"` instead
of a boolean, an intent name that does not exist, or content wrapped in markdown
fences despite being told not to. So `llm.py` carries a full coercion layer —
`_coerce_text`, `_coerce_language`, `_coerce_bool`, `_coerce_intent`,
`_strip_code_fences`, `_infer_language` — before Pydantic validation.

`IntentType` is a closed `Literal` of six values, and anything unrecognised maps
to `unsupported_intent` rather than raising or guessing
(`test_unknown_intent_maps_to_unsupported`). The prompt reinforces the same
default: *"When unsure, prefer unsupported_intent over inventing risky actions."*
Failing to a safe no-op is the correct behaviour for a system holding write
access.

### 4.6 Compound commands with threaded context

"Make a folder called notes, then save a summary in it" is two steps where the
second depends on the first. `IntentResult.steps` holds them in execution order
and `_apply_context` threads state forward: a bare filename after a folder
creation is rewritten relative to that folder, and a `write_text` step with empty
content inherits the previous step's text output.

Both are `model_copy(update=…)` — a new step object, never a mutation of the
plan the user confirmed. Four tests cover this
(`test_compound_folder_then_file_uses_folder_context`,
`test_summary_then_store_text_file`, `test_more_than_two_steps_supported`,
`test_unsupported_intent_is_reported_without_crash`).

### 4.7 Confirmation scoped to what actually mutates

`requires_confirmation` is set only when a step's intent is `create_file`,
`write_code`, or `write_text`. Summaries and chat execute without a prompt.
Confirmation fatigue is a real failure mode — an agent that asks about everything
trains the user to click through, which is worse than not asking.

---

## 5. Results

What the repository demonstrably does:

| | |
|---|---|
| Source files | 10, across `src/voice_agent/` and `tests/` |
| Test functions | **20** — 15 on tools, 5 on intent normalisation |
| Intent types | 6, closed enum |
| LLM providers | 3, interchangeable (Ollama local by default) |
| STT providers | 2 |
| Sandbox escapes possible via path input | none found; blocked by resolve-and-assert |
| C++ written without compiling | none; `_finalize_cpp` raises instead |

The test suite is concentrated where the risk is. Of the 15 tool tests, the ones
that matter most are the path-escape block, the placeholder flattening, and the
two C++ tests that assert a fragment is repaired into something `g++` accepts.

**No performance figures appear here** because the repository contains no
benchmark. Latency is dominated by the STT call and by local model inference,
neither of which is measured.

---

## 6. Limitations

1. **`_cpp_fallback_program` returns a fixed max-element program.** If repair
   fails, the fallback writes a canned program unrelated to the request. It is
   labelled in the return message, but a user skimming output could mistake it
   for their answer. Refusing outright would be more honest than substituting.
2. **Validation is C++-only.** Python, JavaScript, TypeScript, Java, Rust and Go
   are all accepted in `_extension` and written unchecked. A syntax check per
   language — even just `ast.parse` for Python — would close most of the gap.
3. **The sandbox protects paths, not content.** Generated code is never run, so
   this is currently fine; the moment execution is added the threat model needs
   rebuilding from scratch.
4. **Session memory only.** State lives in the Streamlit session and is lost on
   reload.
5. **No test covers the provider call paths.** All 5 LLM tests exercise the
   normalisation helpers; the three `classify` branches are untested, so a
   provider-side API change would surface only at runtime.
6. **`_sanitize_relative_path` flattens anything deeper than two levels**, which
   is a heuristic that will occasionally destroy a legitimate nested path the
   user genuinely wanted.
7. **No structured logging.** Debugging a bad intent means re-running it.

## 7. What I would do next

1. Replace the canned C++ fallback with an explicit refusal (§6.1).
2. Add `ast.parse` validation for Python, matching the C++ ladder.
3. Test the three provider paths against recorded responses.
4. Log transcript → raw JSON → normalised intent → action for every run.
5. Distinguish "flattened a placeholder path" from "flattened your real path" in
   the message shown to the user.

---

## 8. Reproduction

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in keys

# Defaults: local Ollama for intent, Groq for transcription
ollama pull llama3.1:8b

streamlit run app.py
pytest                        # 20 tests
```

Configuration is entirely environment-driven — `STT_PROVIDER`, `LLM_PROVIDER`,
`OLLAMA_MODEL`, `OPENAI_MODEL`, `GROQ_MODEL`, `OUTPUT_DIR`. Every file the agent
creates lands under `OUTPUT_DIR` (default `output/`), and by §4.1 it cannot land
anywhere else.
