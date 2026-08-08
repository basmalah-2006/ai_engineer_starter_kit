import pytest
from pydantic import ValidationError
from service import parse_and_validate, triage_email

class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, messages, model=None):
        self.calls += 1
        return self.responses.pop(0)

def test_valid_json_output():
    fake_llm = FakeLLM(
        [
            '{"intent": "support", "urgency": "high", "summary": "User cannot login."}'
        ]
    )

    result = triage_email(
        "I cannot login to my account since this morning.",
        complete_fn=fake_llm,
    )

    assert result.intent == "support"
    assert result.urgency == "high"
    assert result.summary == "User cannot login."

def test_markdown_json_is_cleaned():
    fake_llm = FakeLLM(
        [
            '```json\n{"intent": "billing", "urgency": "medium", "summary": "Customer asks for invoice."}\n```'
        ]
    )

    result = triage_email(
        "Please send me the invoice for last month.",
        complete_fn=fake_llm,
    )

    assert result.intent == "billing"
    assert result.urgency == "medium"

def test_invalid_intent_is_rejected():
    with pytest.raises(ValidationError):
        parse_and_validate(
            '{"intent": "hack", "urgency": "low", "summary": "Invalid intent test."}'
        )

def test_invalid_urgency_is_rejected():
    with pytest.raises(ValidationError):
        parse_and_validate(
            '{"intent": "support", "urgency": "urgent", "summary": "Invalid urgency test."}'
        )

def test_retry_when_first_output_is_invalid():
    fake_llm = FakeLLM(
        [
            '{"intent": "unknown", "urgency": "low", "summary": "Bad output."}',
            '{"intent": "sales", "urgency": "low", "summary": "Customer asks about pricing."}',
        ]
    )

    result = triage_email(
        "I want to know your pricing plans.",
        complete_fn=fake_llm,
        max_retries=2,
    )

    assert fake_llm.calls == 2
    assert result.intent == "sales"
    assert result.urgency == "low"