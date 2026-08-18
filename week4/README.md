# Private Journaling Assistant — Offline-First AI Tool

A small Python journaling assistant that analyzes daily journal entries and returns a structured reflection containing mood analysis, a daily summary, and advice or a reflective question.

This project demonstrates how to use local LLMs with Ollama to build a privacy-first AI application that can process sensitive journal entries locally without sending them to external servers.

The application also supports an optional hosted mode using the Groq API for higher-quality responses.

## Features

* Runs locally using Llama 3.1 8B through Ollama.
* Keeps journal entries on the user's machine in offline mode.
* Supports English and Arabic journal entries.
* Generates mood analysis, daily summaries, and advice/reflection.
* Supports both local Ollama and hosted Groq API backends.
* Uses the OpenAI Python SDK for a unified interface.
* Uses Rich for formatted terminal output.
* Displays token usage, processing time, and generation speed.
* Uses `.env` for secure API key configuration.
* Handles errors when Ollama is unavailable.
* Validates empty input before sending it to the model.

## AI Response Structure

For every journal entry, the assistant generates three sections:

| Section               | Description                                                        |
| --------------------- | ------------------------------------------------------------------ |
| `Mood Analysis`       | A brief analysis of the user's emotional state                     |
| `Daily Summary`       | A short summary of the day                                         |
| `Advice / Reflection` | A thoughtful piece of advice or a reflective question for tomorrow |

Example output:

```text
Mood Analysis
You seem to have had a challenging but productive day, with some stress balanced by a sense of accomplishment.

Daily Summary
A day of small struggles and meaningful progress.

Advice / Reflection
What is one thing you can let go of tomorrow to make your day feel lighter?
```

## AI Models

### Local Model

**Llama 3.1 8B — 4-bit Quantized**

The local model runs through Ollama and is the default backend.

### Hosted Model

**GPT-OSS 20B** (default) — with **GPT-OSS 120B** available for higher-quality responses.

The hosted model runs through the Groq API and can be enabled when higher-quality responses are preferred.

> **Note:** Groq deprecated `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` on August 16, 2026. This project now uses `openai/gpt-oss-20b` (replacement for the 8B model) and `openai/gpt-oss-120b` (replacement for the 70B model) as the hosted options.

## Benchmark

The local model was selected based on the available hardware and the requirements of the application.

| Metric         | Result                  |
| -------------- | ----------------------- |
| Model          | Llama 3.1 8B            |
| Quantization   | 4-bit                   |
| RAM            | 12 GB                   |
| CPU            | Intel Core i5           |
| GPU            | NVIDIA GeForce RTX 2050 |
| Dedicated VRAM | 4 GB                    |
| Average Speed  | ~9 tokens/sec           |

The benchmark focuses on practical local inference performance for empathetic writing, mood analysis, and summarization.

More details are available in `BENCHMARK.md`.

## Project Structure

```text
week4/
│
├── journal.py
├── BENCHMARK.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/basmalah-2006/ai_engineer_starter_kit.git
cd ai_engineer_starter_kit/week4
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Install Ollama

Download and install Ollama from:

https://ollama.com/download

Pull the required local model:

```bash
ollama pull llama3.1:8b
```

Verify that the model is available:

```bash
ollama list
```

## Environment Variables

Create a `.env` file from the example file.

### Linux / macOS

```bash
cp .env.example .env
```

### Windows

```bash
copy .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_groq_api_key_here
OPENAI_MODEL=openai/gpt-oss-20b
```

The API key is only required when using hosted mode.

If `USE_LOCAL = True`, the application runs locally through Ollama and does not require an API key.

## Running the Application

### Local Mode

Make sure:

```python
USE_LOCAL = True
```

in `journal.py`.

Then run:

```bash
python journal.py
```

The application will display:

```text
LOCAL (Offline & Private)

Write about your day.
```

Write your journal entry and press **Enter twice** or type:

```text
DONE
```

The assistant will then generate the journal analysis.

### Hosted Mode

To use the Groq API:

1. Set:

```python
USE_LOCAL = False
```

2. Add a valid Groq API key to `.env`.
3. Run:

```bash
python journal.py
```

## How It Works

1. The user writes a daily journal entry.
2. The entry is passed to the journal assistant.
3. The application selects either the local Ollama model or the hosted Groq model.
4. A prompt instructs the model to analyze the journal entry.
5. The model generates mood analysis, a daily summary, and advice/reflection.
6. The response is formatted using Rich.
7. Performance statistics are displayed after generation.

## Local Mode

The default mode uses Ollama and processes the journal entry locally.

The application connects to the local OpenAI-compatible Ollama endpoint:

```text
http://localhost:11434/v1
```

In this mode, the journal entry is not sent to an external AI API.

This makes local inference useful for applications where **privacy, offline access, and data control** are important.

## Hosted Mode

The application can switch to a hosted LLM using the Groq API.

The same OpenAI-compatible client interface is used for both backends. Switching between them is controlled by the `USE_LOCAL` configuration flag and the model's `base_url`.

This demonstrates how an application can remain flexible while supporting both local and hosted inference.

## Performance Tracking

The application measures the performance of each model response.

It calculates:

* Total tokens generated.
* Processing time.
* Tokens per second.

Example:

```text
245 tokens | 27.18 seconds | 9.02 tokens/sec
```

These metrics make it possible to evaluate local LLM performance on consumer hardware.

## Prompt Engineering Techniques Used

This project uses several prompt-engineering techniques:

* Clear role definition for the assistant.
* Explicit response sections.
* Structured Markdown output.
* Tone guidance for warm and non-judgmental responses.
* Instructions for empathetic and reflective writing.
* Multilingual awareness.
* Temperature tuning (`temperature=0.8`) for creative responses.

## Reliability and Safety Notes

The project applies several practical reliability techniques:

* Uses an OpenAI-compatible SDK for both local and hosted models.
* Graceful error handling when Ollama is not running.
* Input validation for empty journal entries.
* API keys are loaded from `.env` and never hardcoded.
* Local mode avoids sending private journal entries to external services.
* AI-generated reflections are clearly positioned as personal reflection support rather than professional advice.

## Privacy Considerations

Privacy is a core design goal of this project.

When using local mode:

```text
User Journal
     │
     ▼
Local Python Application
     │
     ▼
Ollama
     │
     ▼
Llama 3.1 8B
```

The journal entry is processed locally on the user's machine.

When hosted mode is enabled, journal content is sent to the configured external API provider for inference.

Users should therefore choose the backend according to their privacy requirements.

## Future Improvements

Possible next steps:

* Save journal entries locally by date.
* Add mood tracking over time.
* Visualize mood trends.
* Add Local RAG for searching previous journal entries.
* Add a Streamlit or Gradio interface.
* Add local encryption for stored journals.
* Add support for additional local models.
* Add automated evaluation of response quality.

## Disclaimer

This project is designed for **personal reflection and journaling support**.

The AI-generated mood analysis and advice should **not** be considered professional psychological or medical advice.

## Week 4 Alignment

This project was built as **Mini Project 4** for the **Helwan Career Center 12-Week Industry Roadmap**, Week 4: **Local LLMs & Open-Source Tooling**.

It demonstrates:

* **LO 4.1** — Installed and ran an open model locally using Ollama and used its OpenAI-compatible endpoint.
* **LO 4.2** — Understood quantization and used a 4-bit local model on hardware with 4GB dedicated VRAM.
* **LO 4.3** — Selected Llama 3.1 8B based on hardware constraints, quality requirements, and privacy needs.
* **LO 4.4** — Switched between hosted and local inference using a configuration flag and different `base_url`.
* **LO 4.5** — Evaluated when local inference is preferable for privacy, cost, and offline usage versus hosted APIs for higher model capability and convenience.

## Author

**Basmalah Ahmed**

AI Engineer Starter Kit — Week 4
