import json
import logging
from openai import OpenAI

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional dependency
    InferenceClient = None

from config import (
    CLAUDE_API_KEY,
    CLAUDE_MODEL,
    DEEPSEEK_API_KEY,
    HF_MODEL_ID,
    HF_MODEL_MAGNUM_ID,
    HF_MODEL_TLAMA_ID,
    HF_TOKEN,
    YANDEX_CLOUD_API_KEY,
    YANDEX_PROJECT_ID,
    YANDEX_PROMPT_ID,
    YANDEX_MODEL_ID,
)

DEFAULT_PROVIDER = "deepseek"
AVAILABLE_PROVIDERS = (
    "deepseek",
    "yandex",
    "claude",
    "huggingface",
    "huggingface-magnum",
    "huggingface-tinyllama",
)
DEEPSEEK_MODEL = "deepseek-chat"
TINYLLAMA_MODEL_ID = HF_MODEL_TLAMA_ID

readonly_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)
_log = logging.getLogger(__name__)

_claude_client: "anthropic.Anthropic | None" = None
_yandex_client: OpenAI | None = None
_hf_client: OpenAI | None = None
_hf_inference_client: "InferenceClient | None" = None
_hf_inference_featherless_client: "InferenceClient | None" = None

def _log_raw_result(provider: str, model: str | None, result: object) -> None:
    model_label = model or "-"
    try:
        _log.info("%s/%s raw result: %r", provider, model_label, result)
    except Exception:
        _log.info("%s/%s raw result: <unprintable>", provider, model_label)

def _normalize_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> dict[str, int]:
    prompt = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0
    completion = (
        int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0
    )
    total = int(total_tokens) if isinstance(total_tokens, (int, float)) else prompt + completion
    if total == 0 and (prompt or completion):
        total = prompt + completion
    return {
        "prompt_tokens": max(prompt, 0),
        "completion_tokens": max(completion, 0),
        "total_tokens": max(total, 0),
    }


def _extract_usage_tokens(usage: object) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None

    def _get(*keys: str) -> int | None:
        for key in keys:
            if isinstance(usage, dict):
                if key in usage:
                    return usage.get(key)
            else:
                value = getattr(usage, key, None)
                if value is not None:
                    return value
        return None

    prompt = _get("prompt_tokens", "input_tokens")
    completion = _get("completion_tokens", "output_tokens")
    total = _get("total_tokens", "total")
    return prompt, completion, total

def _extract_text_generation_result(
    result: object,
) -> tuple[str, int | None, int | None, int | None]:
    if isinstance(result, dict):
        text = (
            (result.get("generated_text") or result.get("text") or "")
            if isinstance(result, dict)
            else ""
        )
        usage = result.get("usage")
        details = result.get("details")
    else:
        text = getattr(result, "generated_text", None) or ""
        usage = getattr(result, "usage", None)
        details = getattr(result, "details", None)

    prompt, completion, total = _extract_usage_tokens(usage)

    if details is not None:
        if isinstance(details, dict):
            completion = completion or details.get("generated_tokens") or details.get(
                "output_tokens"
            )
            prefill = details.get("prefill")
            if isinstance(prefill, list):
                prompt = prompt or len(prefill)
            elif isinstance(prefill, int):
                prompt = prompt or prefill
            else:
                prompt = prompt or details.get("prompt_tokens") or details.get("input_tokens")
        else:
            completion = completion or getattr(details, "generated_tokens", None)
            prefill = getattr(details, "prefill", None)
            if isinstance(prefill, list):
                prompt = prompt or len(prefill)
            else:
                prompt = prompt or getattr(details, "prompt_tokens", None)

    return str(text).strip(), prompt, completion, total

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


def _extract_responses_text(response: object) -> str:
    text_field = getattr(response, "text", None)
    if isinstance(text_field, str) and text_field.strip():
        return text_field.strip()

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                item_text = item.get("text")
                if isinstance(item_text, str) and item_text.strip():
                    texts.append(item_text)
                    continue
                content = item.get("content")
            else:
                item_text = getattr(item, "text", None)
                if isinstance(item_text, str) and item_text.strip():
                    texts.append(item_text)
                    continue
                content = getattr(item, "content", None)
            if not isinstance(content, list):
                if isinstance(content, str):
                    texts.append(content)
                continue
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("output_text")
                else:
                    text = getattr(block, "text", None)
                if text:
                    texts.append(text)
    return "".join(texts).strip()


def _response_debug_snapshot(response: object) -> dict:
    if isinstance(response, dict):
        return {"type": "dict", "keys": list(response.keys())}
    if hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return {"type": type(response).__name__, "keys": list(dumped.keys())}
        except Exception:
            pass
    return {
        "type": type(response).__name__,
        "has_output_text": hasattr(response, "output_text"),
        "has_output": hasattr(response, "output"),
        "has_usage": hasattr(response, "usage"),
    }


def _yandex_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, dict[str, int]]:
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
    model_label = YANDEX_MODEL_ID or (f"prompt:{YANDEX_PROMPT_ID}" if YANDEX_PROMPT_ID else None)
    _log_raw_result("yandex", model_label, response)
    text = _extract_yandex_response_text(response)
    prompt, completion, total = _extract_usage_tokens(getattr(response, "usage", None))
    return text, _normalize_usage(prompt, completion, total)


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


def _get_hf_client() -> OpenAI:
    global _hf_client
    if _hf_client is None:
        _hf_client = OpenAI(
            api_key=HF_TOKEN,
            base_url="https://router.huggingface.co/v1",
        )
    return _hf_client


def _get_hf_inference_client() -> "InferenceClient":
    if InferenceClient is None:
        raise RuntimeError(
            "Hugging Face InferenceClient requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        )
    global _hf_inference_client
    if _hf_inference_client is None:
        _hf_inference_client = InferenceClient(
            provider="auto",
            api_key=HF_TOKEN,
        )
    return _hf_inference_client


def _get_hf_inference_featherless_client() -> "InferenceClient":
    if InferenceClient is None:
        raise RuntimeError(
            "Hugging Face InferenceClient requires huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        )
    global _hf_inference_featherless_client
    if _hf_inference_featherless_client is None:
        _hf_inference_featherless_client = InferenceClient(
            provider="featherless-ai",
            api_key=HF_TOKEN,
        )
    return _hf_inference_featherless_client


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
) -> tuple[str, dict[str, int]]:
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
    _log_raw_result("claude", CLAUDE_MODEL, response)

    texts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))

    text = "".join(texts).strip()
    prompt, completion, total = _extract_usage_tokens(getattr(response, "usage", None))
    return text, _normalize_usage(prompt, completion, total)

def _huggingface_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, dict[str, int]]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN (Hugging Face token) is not configured")

    client = _get_hf_client()
    response = client.chat.completions.create(
        model=HF_MODEL_ID,
        messages=messages,
        temperature=temperature,
    )
    _log_raw_result("huggingface", HF_MODEL_ID, response)
    text = response.choices[0].message.content.strip()
    prompt, completion, total = _extract_usage_tokens(getattr(response, "usage", None))
    return text, _normalize_usage(prompt, completion, total)

def _huggingface_magnum_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, dict[str, int]]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN (Hugging Face token) is not configured")

    try:
        client = _get_hf_client()
        response = client.chat.completions.create(
            model=HF_MODEL_MAGNUM_ID,
            messages=messages,
            temperature=temperature,
        )
        _log_raw_result("huggingface-magnum", HF_MODEL_MAGNUM_ID, response)
        text = response.choices[0].message.content.strip()
        prompt, completion, total = _extract_usage_tokens(getattr(response, "usage", None))
        return text, _normalize_usage(prompt, completion, total)
    except Exception as exc:
        error_text = str(exc).lower()
        if "model_not_supported" not in error_text and "not supported" not in error_text:
            raise

    client = _get_hf_inference_client()
    try:
        response = client.chat_completion(
            model=HF_MODEL_MAGNUM_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        _log_raw_result("huggingface-magnum-inference", HF_MODEL_MAGNUM_ID, response)
        content = response.choices[0].message.get("content", "")
        text = (content or "").strip()
        prompt, completion, total = _extract_usage_tokens(getattr(response, "usage", None))
        return text, _normalize_usage(prompt, completion, total)
    except AttributeError:
        prompt = _plain_text_from_messages(messages)
        if not prompt:
            raise RuntimeError("No content to send to Hugging Face model")
        try:
            result = client.text_generation(
                prompt,
                model=HF_MODEL_MAGNUM_ID,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
                details=True,
            )
        except Exception:
            result = client.text_generation(
                prompt,
                model=HF_MODEL_MAGNUM_ID,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
            )

        _log_raw_result("huggingface-magnum-text_generation", HF_MODEL_MAGNUM_ID, result)
        text, prompt_tokens, completion, total = _extract_text_generation_result(result)
        if prompt_tokens is not None or completion is not None or total is not None:
            return text, _normalize_usage(prompt_tokens, completion, total)
        if text:
            return text, _normalize_usage(0, 0, 0)

        return str(result).strip(), _normalize_usage(0, 0, 0)


def _huggingface_tinyllama_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, dict[str, int]]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN (Hugging Face token) is not configured")

    try:
        client = _get_hf_client()
        prompt = _plain_text_from_messages(messages)
        if not prompt:
            raise RuntimeError("No content to send to Hugging Face model")
        response = client.responses.create(
            model=TINYLLAMA_MODEL_ID,
            input=prompt,
            temperature=temperature,
            max_output_tokens=512,
        )
        _log_raw_result("huggingface-tinyllama-responses", TINYLLAMA_MODEL_ID, response)
        _log.debug("TinyLlama via HF router responses.create")
        _log.debug("TinyLlama responses meta: %s", _response_debug_snapshot(response))
        _log.debug(
            "TinyLlama responses output_text len=%s usage=%s",
            len(getattr(response, "output_text", "") or ""),
            getattr(response, "usage", None),
        )
        text = _extract_responses_text(response)
        if not text.strip():
            raise RuntimeError(
                "TinyLlama недоступна в HF Router (пустой ответ). "
                "Попробуйте позже или выберите другую модель."
            )
        prompt_tokens, completion_tokens, total_tokens = _extract_usage_tokens(
            getattr(response, "usage", None)
        )
        return text, _normalize_usage(prompt_tokens, completion_tokens, total_tokens)
    except Exception:
        pass

    try:
        client = _get_hf_client()
        response = client.chat.completions.create(
            model=TINYLLAMA_MODEL_ID,
            messages=messages,
            temperature=temperature,
        )
        _log_raw_result("huggingface-tinyllama-chat", TINYLLAMA_MODEL_ID, response)
        _log.debug("TinyLlama via HF router chat.completions")
        text = response.choices[0].message.content.strip()
        prompt_tokens, completion_tokens, total_tokens = _extract_usage_tokens(
            getattr(response, "usage", None)
        )
        return text, _normalize_usage(prompt_tokens, completion_tokens, total_tokens)
    except Exception:
        prompt = _plain_text_from_messages(messages)
        if not prompt:
            raise RuntimeError("No content to send to Hugging Face model")

    def _text_generation_with_client(
        client: "InferenceClient",
        client_label: str,
    ) -> tuple[str, dict[str, int]]:
        _log.debug("TinyLlama via HF Inference fallback (%s)", client_label)

        def _chat_completion_fallback() -> tuple[str, dict[str, int]]:
            response = client.chat_completion(
                model=TINYLLAMA_MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=512,
            )
            _log_raw_result(
                f"tinyllama-chat_completion-{client_label}", TINYLLAMA_MODEL_ID, response
            )
            content = response.choices[0].message.get("content", "")
            text = (content or "").strip()
            prompt_tokens, completion, total = _extract_usage_tokens(
                getattr(response, "usage", None)
            )
            if not text:
                raise RuntimeError(
                    "TinyLlama недоступна в HF Inference (пустой ответ). "
                    "Попробуйте позже или выберите другую модель."
                )
            return text, _normalize_usage(prompt_tokens, completion, total)

        try:
            result = client.text_generation(
                prompt,
                model=TINYLLAMA_MODEL_ID,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
                details=True,
                decoder_input_details=True,
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if "not supported for task text-generation" in error_text or "supported task: conversational" in error_text:
                return _chat_completion_fallback()
            result = client.text_generation(
                prompt,
                model=TINYLLAMA_MODEL_ID,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False,
            )

        _log_raw_result(f"tinyllama-text_generation-{client_label}", TINYLLAMA_MODEL_ID, result)
        text, prompt_tokens, completion, total = _extract_text_generation_result(result)
        if not text:
            return _chat_completion_fallback()
        if prompt_tokens is not None or completion is not None or total is not None:
            return text, _normalize_usage(prompt_tokens, completion, total)
        return text, _normalize_usage(0, 0, 0)

    try:
        return _text_generation_with_client(_get_hf_inference_client(), "auto")
    except Exception:
        return _text_generation_with_client(_get_hf_inference_featherless_client(), "featherless-ai")


def _deepseek_completion(
    messages: list[dict[str, str]],
    temperature: float,
) -> tuple[str, dict[str, int]]:
    response = readonly_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
    )
    _log_raw_result("deepseek", DEEPSEEK_MODEL, response)
    text = response.choices[0].message.content.strip()
    prompt, completion, total = _extract_usage_tokens(getattr(response, "usage", None))
    return text, _normalize_usage(prompt, completion, total)


def chat_completion(
    messages: list[dict[str, str]],
    provider: str | None = None,
    temperature: float = 0.6,
) -> tuple[str, dict[str, int]]:
    provider = provider or DEFAULT_PROVIDER
    if provider == "deepseek":
        return _deepseek_completion(messages, temperature)
    if provider == "yandex":
        return _yandex_completion(messages, temperature)
    if provider == "claude":
        return _claude_completion(messages, temperature)
    if provider == "huggingface":
        return _huggingface_completion(messages, temperature)
    if provider == "huggingface-magnum":
        return _huggingface_magnum_completion(messages, temperature)
    if provider == "huggingface-tinyllama":
        return _huggingface_tinyllama_completion(messages, temperature)
    raise RuntimeError(f"Unknown provider: {provider}")
