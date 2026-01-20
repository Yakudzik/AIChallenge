import json
import re
from ai_client import chat_completion, DEFAULT_PROVIDER
from prompts import (
    SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    MATHEMATICIAN_PROMPT,
    PHILOSOPHER_PROMPT,
    CREATIVE_PROMPT,
    REFEREE_PROMPT,
)


CLARIFY_STATE_KEY = "clarify_state"
JSON_MODE_KEY = "json_mode"
JSON_MODE_PRETTY = "pretty"
JSON_MODE_CLEAN = "clean"
JSON_MODE_OFF = "off"
DISCUSSION_MODE_KEY = "discussion_mode"
DISCUSSION_MEMORY_KEY = "discussion_memory"
AI_PROVIDER_KEY = "ai_provider"
DEFAULT_LANGUAGE_CODE = "ru"
DEEPSEEK_TEMPERATURE = 0.6
YANDEX_TEMPERATURE = 0.6
CLAUDE_TEMPERATURE = 0.6
DISCUSSION_TEMPERATURE = 0.5
REFEREE_TEMPERATURE = 0.5


def _temperature_for_provider(provider: str) -> float:
    if provider == "yandex":
        return YANDEX_TEMPERATURE
    if provider == "claude":
        return CLAUDE_TEMPERATURE
    return DEEPSEEK_TEMPERATURE


def normalize_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+[\).\-\:]?\s*", "", line)
        line = re.sub(r"^[\-–•]+\s*", "", line)
        if line:
            lines.append(line)
    return lines


def generate_next_question(
    original: str,
    qas: list[dict[str, str]],
    asked: list[str],
    provider: str = DEFAULT_PROVIDER,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = _temperature_for_provider(DEFAULT_PROVIDER),
) -> str | None:
    response_text = chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Исходный запрос: {original}\n"
                    f"Диалог уточнений (вопрос/ответ): {json.dumps(qas, ensure_ascii=False)}\n"
                    f"Уже заданные вопросы: {json.dumps(asked, ensure_ascii=False)}"
                ),
            },
        ],
        provider=provider,
        temperature=temperature,
    )
    raw = response_text.strip()
    if not raw:
        return None
    normalized = normalize_lines(raw)
    if not normalized:
        normalized = [raw.strip()]
    combined = "\n".join(normalized).strip()
    if combined.lower() in {"нет", "не нужно", "достаточно", "без вопросов"}:
        return None
    return combined


def summarize_with_answers(
    original: str,
    answers: list[str],
    provider: str = DEFAULT_PROVIDER,
    temperature: float = _temperature_for_provider(DEFAULT_PROVIDER),
) -> str:
    response_text = chat_completion(
        messages=[
            {
                "role": "system",
                "content": SUMMARY_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Исходный запрос: {original}\n"
                    f"Уточнения: {json.dumps(answers, ensure_ascii=False)}"
                ),
            },
        ],
        provider=provider,
        temperature=temperature,
    )
    return response_text.strip()


def generate_role_answer(
    system_prompt: str,
    text: str,
    provider: str = DEFAULT_PROVIDER,
    temperature: float = DISCUSSION_TEMPERATURE,
) -> str:
    response_text = chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        provider=provider,
        temperature=temperature,
    )
    return response_text.strip()


def _discussion_prompt_for_provider(provider: str) -> tuple[str, str]:
    if provider == "yandex":
        return "Философ", PHILOSOPHER_PROMPT
    if provider == "claude":
        return "Креативщик", CREATIVE_PROMPT
    return "Математик", MATHEMATICIAN_PROMPT


def _discussion_provider_pairs() -> list[tuple[str, str]]:
    return [
        (DEFAULT_PROVIDER, "Математик"),
        ("yandex", "Философ"),
        ("claude", "Креативщик"),
    ]


def generate_discussion_answers(
    text: str,
    temperature_by_provider: dict[str, float] | None = None,
) -> list[tuple[str, str, str]]:
    answers = []
    for provider, label in _discussion_provider_pairs():
        _, prompt = _discussion_prompt_for_provider(provider)
        temperature = (temperature_by_provider or {}).get(
            provider, _temperature_for_provider(provider)
        )
        answers.append(
            (provider, label, generate_role_answer(prompt, text, provider, temperature))
        )
    return answers


def generate_referee_answer(
    discussion_memory: dict[str, str],
    temperature: float = _temperature_for_provider(DEFAULT_PROVIDER),
) -> str:
    response_text = chat_completion(
        messages=[
            {
                "role": "system",
                "content": REFEREE_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(discussion_memory, ensure_ascii=False),
            },
        ],
        provider=DEFAULT_PROVIDER,
        temperature=temperature,
    )
    return response_text.strip()


def format_discussion(answers: dict[str, str]) -> str:
    blocks = []
    for role, content in answers.items():
        content = content.strip() or "..."
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks)
