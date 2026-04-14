from pathlib import Path
import tempfile

import streamlit as st

from src.voice_agent.agent import VoiceAgent
from src.voice_agent.config import AppConfig
from src.voice_agent.models import AgentResult


st.set_page_config(
    page_title="Voice Agent Builder",
    layout="wide",
)


def _save_uploaded_audio(audio_bytes: bytes, suffix: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "voice_agent_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"audio_input{suffix}"
    temp_file.write_bytes(audio_bytes)
    return temp_file


def _init_state() -> None:
    if "pending_result" not in st.session_state:
        st.session_state.pending_result = None
    if "history" not in st.session_state:
        st.session_state.history = []


def _render_result(result: AgentResult, title: str = "Latest Run") -> None:
    st.subheader(title)
    left, right = st.columns(2)
    detected_intents = result.intent.intents or [result.intent.primary_intent]

    with left:
        st.markdown("**Transcription**")
        st.write(result.transcript)
        st.markdown("**Detected Intent**")
        st.write(", ".join(detected_intents))
        st.markdown("**Action Taken**")
        st.write(result.action_summary)

    with right:
        st.markdown("**Final Output**")
        st.write(result.output)
        if result.created_paths:
            st.markdown("**Created / Updated Paths**")
            for path in result.created_paths:
                st.code(str(path))


def _resolve_audio_source() -> tuple[bytes | None, str, str | None]:
    source_choice = st.radio(
        "Choose input source",
        options=["Upload audio file", "Record with microphone"],
        horizontal=True,
    )

    if source_choice == "Upload audio file":
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "m4a"],
            accept_multiple_files=False,
        )
        if uploaded_file is None:
            return None, ".wav", None

        source_bytes = uploaded_file.read()
        suffix = Path(uploaded_file.name).suffix or ".wav"
        st.audio(source_bytes)
        return source_bytes, suffix, uploaded_file.name

    microphone_audio = st.audio_input("Record with your microphone")
    if microphone_audio is None:
        return None, ".wav", None

    source_bytes = microphone_audio.read()
    st.audio(source_bytes, format="audio/wav")
    return source_bytes, ".wav", "microphone_recording.wav"


def main() -> None:
    _init_state()
    config = AppConfig.from_env()
    agent = VoiceAgent(config=config)

    st.title("Voice-Controlled Local AI Agent")
    st.caption(
        "Upload or record audio, transcribe it, detect intent, and safely execute actions inside the output folder."
    )

    with st.sidebar:
        st.header("Settings")
        st.write(f"Output directory: `{config.output_dir}`")
        st.write(f"STT provider: `{config.stt_provider}`")
        st.write(f"LLM provider: `{config.llm_provider}`")
        st.checkbox(
            "Require confirmation for file operations",
            value=True,
            disabled=True,
            help="Enabled in this build to keep local actions safe.",
        )

    source_bytes, suffix, source_name = _resolve_audio_source()
    if source_name:
        st.caption(f"Active input: `{source_name}`")

    if st.button("Run Agent", type="primary", disabled=source_bytes is None):
        try:
            audio_path = _save_uploaded_audio(source_bytes, suffix)
            result = agent.prepare(audio_path)
            st.session_state.pending_result = result
        except Exception as exc:
            st.error(f"Agent run failed: {exc}")

    pending_result: AgentResult | None = st.session_state.pending_result
    if pending_result is not None:
        _render_result(pending_result, title="Pending Confirmation")
        if pending_result.requires_confirmation:
            if st.button("Confirm Action"):
                try:
                    final_result = agent.execute(pending_result)
                    st.session_state.history.insert(0, final_result)
                    st.session_state.pending_result = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Execution failed: {exc}")
        else:
            try:
                final_result = agent.execute(pending_result)
                st.session_state.history.insert(0, final_result)
                st.session_state.pending_result = None
                st.rerun()
            except Exception as exc:
                st.error(f"Execution failed: {exc}")

    if st.session_state.history:
        st.divider()
        st.header("Session History")
        for idx, item in enumerate(st.session_state.history, start=1):
            _render_result(item, title=f"Run {idx}")


if __name__ == "__main__":
    main()
