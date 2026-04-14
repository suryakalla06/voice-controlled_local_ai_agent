# Voice-Controlled Local AI Agent

This project implements the assignment from `Mem0_ AI_ML & Generative AI Developer Intern Assignment.pdf`: a voice-driven AI agent that accepts audio, transcribes speech, classifies the user's intent, safely executes local actions inside `output/`, and shows the full pipeline in a Streamlit UI.

## Assignment status

Requirement-by-requirement status against the PDF:

- Audio input from microphone: satisfied
- Audio file upload: satisfied
- Speech-to-text: satisfied through OpenAI or Groq API-based STT
- Local or API STT note in README: satisfied
- Intent understanding with LLM: satisfied through Ollama, OpenAI, or Groq
- Minimum supported intents:
  - create file: satisfied
  - write code to new or existing file: satisfied
  - summarize text: satisfied
  - general chat: satisfied
- Tool execution for local file operations: satisfied
- Create files or folders inside sandboxed `output/`: satisfied
- Code generation saved directly to file: satisfied
- Text summarization: satisfied
- UI shows transcription: satisfied
- UI shows detected intent: satisfied
- UI shows action taken: satisfied
- UI shows final output/result: satisfied
- Safety constraint restricting file writes to `output/`: satisfied
- Human-in-the-loop confirmation for file operations: implemented as a bonus
- Session history / memory within session: implemented as a bonus

Items not included in the codebase because they are submission deliverables rather than runtime features:

- Public GitHub repository link
- 2 to 3 minute demo video
- Technical article
- Final form submission

## What this app does

- Accepts audio from either:
  - microphone recording inside the Streamlit UI
  - uploaded audio files such as `.wav`, `.mp3`, and `.m4a`
- Transcribes speech to text
- Detects one of the supported intents:
  - create file
  - create folder
  - write code
  - summarize text
  - general chat
- Executes file actions only inside the `output/` directory
- Shows:
  - transcription
  - detected intent
  - action taken
  - final output
- Requires confirmation before file-writing operations
- Stores session history in the current UI session

## Project structure

- `app.py`: Streamlit frontend
- `src/voice_agent/config.py`: environment/config loading
- `src/voice_agent/stt.py`: speech-to-text provider wrapper
- `src/voice_agent/llm.py`: intent classification provider wrapper
- `src/voice_agent/tools.py`: safe local tool execution
- `src/voice_agent/agent.py`: end-to-end orchestration
- `output/`: sandboxed folder for generated files
- `tests/`: automated edge-case tests

## Before you start

You need:

- Python `3.10+`
- `pip`
- A terminal opened in this project folder
- At least one working STT provider
- At least one working LLM provider

This app supports these providers:

- STT:
  - OpenAI
  - Groq
- LLM:
  - Ollama
  - OpenAI
  - Groq

## Recommended setup paths

Choose one of these before running the app.

### Option 1: easiest overall

Use APIs for both STT and LLM.

- `STT_PROVIDER=openai`
- `LLM_PROVIDER=openai`
- requires:
  - `OPENAI_API_KEY`

### Option 2: local-ish setup

Use OpenAI for speech-to-text and Ollama for intent classification.

- `STT_PROVIDER=openai`
- `LLM_PROVIDER=ollama`
- requires:
  - `OPENAI_API_KEY`
  - Ollama running locally
  - model pulled locally, for example `llama3.1:8b`

### Option 3: Groq-based setup

Use Groq where possible.

- `STT_PROVIDER=groq`
- `LLM_PROVIDER=groq`
- requires:
  - `GROQ_API_KEY`

## Exact setup steps

Follow these steps in order.

### 1. Open the project folder

If you are using WSL:

```bash
cd /home/laksh/projects/overbase
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

If `python3` does not work, try:

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Linux / WSL / macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, your terminal should show something like `(.venv)`.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create your `.env` file

On Linux / WSL / macOS:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### 6. Fill in `.env`

Start from this template:

```env
OPENAI_API_KEY=
GROQ_API_KEY=
STT_PROVIDER=openai
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OPENAI_MODEL=gpt-4o-mini
GROQ_MODEL=llama-3.1-8b-instant
OUTPUT_DIR=output
```

Now choose one provider combination and set the variables.

#### Example A: OpenAI for both STT and LLM

```env
OPENAI_API_KEY=your_openai_key_here
STT_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OUTPUT_DIR=output
```

#### Example B: OpenAI STT + local Ollama LLM

```env
OPENAI_API_KEY=your_openai_key_here
STT_PROVIDER=openai
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OUTPUT_DIR=output
```

#### Example C: Groq for both STT and LLM

```env
GROQ_API_KEY=your_groq_key_here
STT_PROVIDER=groq
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
OUTPUT_DIR=output
```

## Extra step only if you use Ollama

If `LLM_PROVIDER=ollama`, do this before starting the app.

### 1. Install Ollama

Install it from the official site:

- [Ollama](https://ollama.com/)

### 2. Start Ollama

In most setups, installing Ollama also starts the local server. If not, start it manually.

### 3. Pull the model

```bash
ollama pull llama3.1:8b
```

### 4. Confirm Ollama is running

Open this in a browser:

- [http://localhost:11434](http://localhost:11434)

Or check from terminal:

```bash
curl http://localhost:11434/api/tags
```

If you get a response, Ollama is reachable.

## Run the app

Make sure:

- your virtual environment is activated
- your `.env` file exists
- your provider keys are set
- Ollama is running if you selected `LLM_PROVIDER=ollama`

Then run:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

- `http://localhost:8501`

Open that URL in your browser.

## Run the automated tests

To verify the app’s edge-case handling locally:

```bash
python -m unittest discover -s tests -v
```

## How to use the app

### Microphone flow

1. Open the app in the browser.
2. Use the microphone recorder in the UI.
3. Record your command.
4. Click `Run Agent`.
5. Review the transcription, detected intent, and planned action.
6. If the action creates or writes a file or folder, click `Confirm Action`.
7. Check the final result in the UI.
8. Look in `output/` for generated files or folders.

### File upload flow

1. Open the app in the browser.
2. Upload an audio file.
3. Click `Run Agent`.
4. Review the transcription, intent, and action.
5. Confirm if required.
6. Check the result in the UI and in `output/`.

## Example prompts to test

You can test the app using audio commands like:

- `Create a file called notes.txt`
- `Create a folder called project_docs`
- `Create a Python file called retry.py with a retry decorator`
- `Create a C plus plus file called max_element.cpp with code to find the maximum element in an array`
- `Summarize this text: Python is a high-level programming language used for web apps, AI, automation, and scripting.`
- `What can you do for me?`

## Where generated files go

All created or updated files and folders are restricted to:

```text
output/
```

This is intentional for safety. The app will reject paths that try to escape this directory.

## Common issues and fixes

### Error: `OPENAI_API_KEY is required`

Cause:

- you selected `openai` as a provider but did not set `OPENAI_API_KEY`

Fix:

- add your key to `.env`
- restart the app

### Error: `GROQ_API_KEY is required`

Cause:

- you selected `groq` as a provider but did not set `GROQ_API_KEY`

Fix:

- add your key to `.env`
- restart the app

### Error connecting to Ollama

Cause:

- Ollama is not running
- the local URL is wrong
- the model has not been pulled

Fix:

- run `ollama pull llama3.1:8b`
- make sure Ollama is running
- confirm `OLLAMA_BASE_URL=http://localhost:11434`

### `streamlit: command not found`

Cause:

- virtual environment is not activated
- dependencies were not installed

Fix:

- activate `.venv`
- run `pip install -r requirements.txt`

### The app opens but fails when I click `Run Agent`

Cause:

- missing API key
- Ollama not running
- invalid audio file

Fix:

- check `.env`
- check provider choice
- check that your uploaded file is a valid audio file
- read the exact error shown in the Streamlit UI

## Notes on local vs API models

The assignment prefers local models. This implementation supports local intent classification through Ollama. Speech-to-text currently uses API-based providers because local Whisper-class models can be heavy on CPU/RAM for modest hardware. If you submit with API STT, mention that hardware tradeoff in your final README/article/demo.

## Quick run checklist

Before running, confirm all of these:

- you are inside the project directory
- `.venv` exists
- `.venv` is activated
- dependencies are installed
- `.env` exists
- provider keys are filled in
- if using Ollama, the server is running and the model is pulled
- you start the app with `streamlit run app.py`

## Deliverables still to complete

After the app is working, you still need:

- a public GitHub repo
- a 2 to 3 minute demo video
- a technical article
- submission through the official form
