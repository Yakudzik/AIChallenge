import json
from openai import OpenAI

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

from config import (
    CLAUDE_API_KEY,
    CLAUDE_MODEL,
    DEEPSEEK_API_KEY,
    YANDEX_CLOUD_API_KEY,
    YANDEX_PROJECT_ID,
    YANDEX_PROMPT_ID,
    YANDEX_MODEL_ID,
)

DEFAULT_PROVIDER = "deepseek"
AVAILABLE_PROVIDERS = ("deepseek", "yandex", "claude")
DEEPSEEK_MODEL = "deepseek-chat"

readonly_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

_claude_client: "anthropic.Anthropic | None" = None
_yandex_client: OpenAI | None = None

def _plain_text_from_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        parts.append(f"{role}: {text}")
    return "\n".join(parts).strip()


def _get_yandex_client() -> OpenAI:
    global _yandex_client
    if _yandex_client is None:
        _yandex_client = OpenAI(
            api_key=YANDEX_CLOUD_API_KEY,
            base_url="https://rest-assistant.api.cloud.yandex.net/v1",
            project=YANDEX_PROJECT_ID,
        )
    return _yandex_client


def _extract_yandex_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)
    return "".join(texts).strip()


def _yandex_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    if not YANDEX_CLOUD_API_KEY or not YANDEX_PROJECT_ID:
        raise RuntimeError("YANDEX_CLOUD_API_KEY/YANDEX_PROJECT_ID is not configured")
    if not (YANDEX_PROMPT_ID or YANDEX_MODEL_ID):
        raise RuntimeError("YANDEX_PROMPT_ID or YANDEX_MODEL_ID is not configured")

    input_text = _plain_text_from_messages(messages)
    if not input_text:
        raise RuntimeError("No content to send to Yandex")

    payload = {
        "input": input_text,
        "temperature": temperature,
    }
    if YANDEX_PROMPT_ID:
        payload["prompt"] = {"id": YANDEX_PROMPT_ID}
    if YANDEX_MODEL_ID:
        payload["model"] = YANDEX_MODEL_ID

    client = _get_yandex_client()
    response = client.responses.create(**payload)
    return _extract_yandex_response_text(response)


def _get_claude_client() -> "anthropic.Anthropic":
    if anthropic is None:
        raise RuntimeError(
            "Claude client requires the anthropic package. "
            "Install it with: pip install anthropic"
        )
    global _claude_client
    if _claude_client is None:
        _claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return _claude_client


def _claude_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str | None]:
    system_parts: list[str] = []
    conversation: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if role == "system":
            system_parts.append(text)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        conversation.append({"role": role, "content": text})

    system_text = "\n\n".join(system_parts).strip() if system_parts else None
    return conversation, system_text


def _claude_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    if not CLAUDE_API_KEY:
        raise RuntimeError("CLAUDE_API_KEY is not configured")

    conversation, system_text = _claude_messages(messages)
    if not conversation:
        raise RuntimeError("No content to send to Claude")

    payload = {
        "model": CLAUDE_MODEL,
        "messages": conversation,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if system_text:
        payload["system"] = system_text

    client = _get_claude_client()
    response = client.messages.create(**payload)

    texts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))

    return "".join(texts).strip()


def _deepseek_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    response = readonly_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def chat_completion(
    messages: list[dict[str, str]],
    provider: str | None = None,
    temperature: float = 0.6,
) -> str:
    provider = provider or DEFAULT_PROVIDER
    if provider == "deepseek":
        return _deepseek_completion(messages, temperature)
    if provider == "yandex":
        return _yandex_completion(messages, temperature)
    if provider == "claude":
        return _claude_completion(messages, temperature)
    raise RuntimeError(f"Unknown provider: {provider}")
