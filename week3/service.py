import json
import re
from pathlib import Path
from pydantic import ValidationError
from schemas import EmailTriage

PROMPT_PATH = Path(__file__).parent / "prompts" / "email_triage.txt"

def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8").strip()

def extract_json(text):
    """
    Sometimes LLMs return JSON inside markdown code fences,
    for example:

    ```json
    { ... }
    ```

    This function tries to extract the JSON part.
    """
    text = text.strip()

    fence_match = re.match(
        r"^```(?:json)?\s*(.*?)\s*```$",
        text,
        re.DOTALL,
    )

    if fence_match:
        return fence_match.group(1)

    json_match = re.search(r"\{.*\}", text, re.DOTALL)

    if json_match:
        return json_match.group(0)

    return text

def parse_and_validate(raw_output):
    json_text = extract_json(raw_output)
    data = json.loads(json_text)
    return EmailTriage.model_validate(data)

def build_messages(raw_text):
    system_prompt = load_prompt()

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": raw_text,
        },
    ]

def default_complete(messages, model=None):
    import os
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key or api_key.startswith("put_"):
        from fake_llm import fake_complete
        return fake_complete(messages, model)

    base_url = os.getenv("OPENAI_BASE_URL")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    model = model or os.getenv("OPENAI_MODEL", "gemini-2.0-flash")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=300,
    )

    return response.choices[0].message.content

def triage_email(raw_text, complete_fn=None, model=None, max_retries=2):
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text is empty")

    complete_fn = complete_fn or default_complete

    messages = build_messages(raw_text)

    last_error = None

    for _ in range(max_retries):
        raw_output = complete_fn(messages, model)

        try:
            return parse_and_validate(raw_output)

        except (json.JSONDecodeError, ValidationError) as error:
            last_error = error

            messages = messages + [
                {
                    "role": "assistant",
                    "content": raw_output,
                },
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid. "
                        f"Error: {error}. "
                        "Return ONLY corrected valid JSON. "
                        "No markdown. No explanations."
                    ),
                },
            ]

    raise ValueError(
        f"LLM output was not valid after {max_retries} attempts. "
        f"Last error: {last_error}"
    )