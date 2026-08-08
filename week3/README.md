# Prompt-Powered Microservice — Customer Email Triage

A small Python microservice that takes raw customer email text and returns clean, validated JSON using an LLM abstraction.

This project demonstrates how to make LLM outputs reliable enough to be used inside real software by enforcing a strict output schema and validating it with Pydantic.

## Features

- Uses a system prompt to force structured JSON output.
- Validates LLM output using Pydantic.
- Extracts JSON even if the model wraps it in markdown code fences.
- Retries automatically if the LLM returns invalid JSON.
- Supports offline mode using a mock LLM when no API key is available.
- Includes 5 passing pytest test cases.

## Output Schema

The service returns JSON with the following fields:

| Field | Type | Allowed Values | Description |
| --- | --- | --- | --- |
| `intent` | string | `support`, `sales`, `billing`, `general`, `spam` | The main intent of the customer message |
| `urgency` | string | `low`, `medium`, `high` | How urgent the customer request is |
| `summary` | string | free text | A short factual summary of the message |

Example output:

```json
{
  "intent": "billing",
  "urgency": "high",
  "summary": "The customer's subscription payment failed and needs urgent resolution."
}
```

## Project Structure

```text
week3/
│
├── prompts/
│   └── email_triage.txt
│
├── tests/
│   └── test_service.py
│
├── main.py
├── schemas.py
├── service.py
├── fake_llm.py
├── requirements.txt
├── README.md
├── .env.example
└── conftest.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/basmalah-2006/ai_engineer_starter_kit.git
cd ai_engineer_starter_kit/week3
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
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

If `OPENAI_API_KEY` is empty, the project automatically runs in offline mode using a deterministic mock LLM.

## Running the Service

Run the example script:

```bash
python main.py
```

The application will send a sample customer email through the LLM layer and print validated JSON output.

## Running Tests

Run the test suite:

```bash
python -m pytest -q
```

Expected result:

```text
5 passed
```

## How It Works

1. Raw customer email text is passed to the service.
2. A system prompt instructs the LLM to return JSON only.
3. The LLM layer returns a response.
4. The service extracts JSON if the response is wrapped in markdown.
5. Pydantic validates the JSON against the expected schema.
6. If the output is invalid, the service retries automatically.
7. A clean, validated structured object is returned.

## Validation Layer

The most important part of this project is the validation layer.

LLMs can sometimes return:

- invalid JSON,
- markdown-wrapped JSON,
- missing fields,
- extra fields,
- invalid enum values,
- hallucinated output.

This service uses Pydantic to enforce the exact expected schema before allowing the output to be used by the application.

This makes the LLM output safer and more suitable for real software systems.

## Offline Mode

This project supports an offline mock LLM for demonstration and testing.

- If `OPENAI_API_KEY` is provided, the service can call a real OpenAI-compatible API.
- If no API key is provided, the service uses `fake_llm.py` to generate deterministic valid JSON.

This allows the validation layer, prompt structure, and tests to be demonstrated without requiring an API key.

## Prompt Engineering Techniques Used

This project uses several prompt-engineering patterns:

- Clear role definition.
- Strict output format instructions.
- Explicit JSON schema.
- Allowed enum values.
- Guardrail against prompt injection.
- Instruction to return JSON only.

## Reliability and Safety Notes

The project applies several practical reliability techniques:

- `temperature=0` for more deterministic output.
- Limited `max_tokens` to reduce cost and latency.
- Strict JSON schema.
- Pydantic validation.
- Automatic retry on invalid output.
- Basic prompt guardrail against instruction manipulation.

## Future Improvements

Possible next steps:

- Add support for multiple LLM providers.
- Add FastAPI endpoint to expose the microservice over HTTP.
- Add evaluation dataset to measure classification accuracy.
- Add stronger prompt-injection defenses.
- Add logging and monitoring.
- Add async API calls.
- Add support for more languages.

## Week 3 Alignment

This project was built as **Mini Project 3** for the **Helwan Career Center 12-Week Industry Roadmap**, Week 3: **LLMs Demystified + Prompt Engineering**.

It demonstrates:

- Prompt engineering using clear instructions and output schemas.
- Structured JSON output.
- Validation of LLM output using Pydantic.
- Basic mitigation of unreliable LLM behavior.
- A reusable module that can plug into real software.

## Author

**Basmalah Ahmed**  
AI Engineer Starter Kit — Week 3