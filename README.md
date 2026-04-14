Items not included in the codebase because they are submission deliverables rather than runtime features:

- 2 to 3 minute demo video
- Final form submission

# Voice-Controlled Local AI Agent

A local AI agent that accepts voice input, understands user intent using LLMs, and executes real actions like file creation, code generation, summarization, and general chat — all through a clean UI.

---

## Features

- Voice Input (supports both microphone and audio file upload )
- Intent Detection using LLM (Ollama - Local)
- Speech-to-Text using Groq API (openai is also supported !)
- Tool Execution (only in the output folder !) :
  - Create files/folders
  - Write code to files
  - Summarize text
  - General chat
- Human-in-the-loop confirmation before file operations (LLM asks permission to confirm the operation)
- Session memory (persists until app reload)
- Streamlit UI displays :
  - Transcription
  - Intent
  - Action
  - Output





---




## Project structure

- `app.py`: Streamlit frontend
- `src/voice_agent/config.py`: environment/config loading
- `src/voice_agent/stt.py`: speech-to-text provider wrapper
- `src/voice_agent/llm.py`: intent classification provider wrapper
- `src/voice_agent/tools.py`: safe local tool execution
- `src/voice_agent/agent.py`: end-to-end orchestration
- `output/`: sandboxed folder for generated files
- `tests/`: automated edge-case tests

---

## Architecture


Audio Input → STT (Groq) → LLM (Ollama) → Intent JSON → Tool Execution → UI Output

---

## Before you start

You need:

- Python `3.10+`
- `pip`
- A terminal opened in this project folder
- At least one working STT provider
- At least one working LLM provider

Note : I am using linux terminal (wsl) and if you are working on windows or mac try to find suitable
       commands.
 

---

This app supports these providers:

- STT:
  - OpenAI
  - Groq
- LLM:
  - Ollama
  - OpenAI
  - Groq

## setup instructions for STT and LLM

Choose one of the providers above before setting to move forward smoothly.

Note : don't use .env.example directly as it results to fail to run the app !  
       copy the contents of the `.env.example` in to a new file `.env` and place your
       secret api_keys in the required places in the `.env` file.

### option 1: easiest overall

Use APIs for both STT and LLM.

- `STT_PROVIDER=openai` (similarly for `groq`)
- `LLM_PROVIDER=openai`
- requires:
  - `OPENAI_API_KEY` (or `GROQ_API_KEY` )

### Option 2: local-ish setup

Use OpenAI for speech-to-text and Ollama for intent classification.

- `STT_PROVIDER=openai` (or `groq`)
- `LLM_PROVIDER=ollama`
- requires:
  - `OPENAI_API_KEY` (or `GROQ_API_KEY` )
  - Ollama running locally
  - model pulled locally, for example `llama3.1:8b`

---


## Setup Instructions

### 1. Create Virtual Environment (Conda)

Note : if you don't have conda in your PC use `python3 -m venv .venv` and activate it
        `source .venv/bin/activate` or you can install it by following the official documentation.

# if you have conda proceed below steps
      
```bash
conda create -n voice-agent python=3.10
```

```bash
conda activate voice-agent
```

`you should be able to see (voice-agent) or (.venv) at the left of your terminal instead of (base) or nothing.`


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your `.env` file


```bash
cp .env.example .env
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


## Where generated files go

All created or updated files and folders are restricted to:

```text
output/
```

This is intentional for safety. The app will reject paths that try to escape this directory.


## Quick run checklist

Before running, confirm all of these:

- you are inside the project directory
- `voice-agent` exists (or `.venv`)
- `voice-agent` is activated
- dependencies are installed
- `.env` exists
- provider keys are filled in
- if using Ollama, the server is running and the model is pulled
- you start the app with `streamlit run app.py`


## Notes on local vs API models

The assignment prefers local models. This implementation supports local intent classification through Ollama. Speech-to-text currently uses API-based providers because local Whisper-class models can be heavy on CPU/RAM for modest hardware.Ollama is used for local inference to ensure privacy and avoid API costs, although cloud-based models may offer lower latency

---