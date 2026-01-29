import time
import logging
import json
import uuid
import re
from datetime import datetime, timedelta, timezone
from message_logic import (
    AI_PROVIDER_KEY,
    CLARIFY_STATE_KEY,
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_PROVIDER,
    DISCUSSION_MODE_KEY,
    DISCUSSION_MEMORY_KEY,
    JSON_MODE_KEY,
    JSON_MODE_PRETTY,
    JSON_MODE_CLEAN,
    JSON_MODE_OFF,
    generate_next_question,
    summarize_with_answers,
    generate_discussion_answers,
    generate_referee_answer,
)
from prompts import (
    SYSTEM_PROMPT,
    SIMPLE_PROMT,
    MATHEMATICIAN_PROMPT,
    PHILOSOPHER_PROMPT,
    CREATIVE_PROMPT,
)
from config import HF_MODEL_ID, HF_MODEL_MAGNUM_ID, HF_MODEL_TLAMA_ID

SYSTEM_PROMPT_KEY = "system_prompt"
TEMPERATURE_KEY = "temperature_by_provider"

_DEFAULT_TEMPERATURES = {
    "deepseek": 0.6,
    "yandex": 0.6,
    "claude": 0.6,
    "huggingface": 0.6,
    "huggingface-magnum": 0.6,
    "huggingface-tinyllama": 0.6,
}

_TEMPERATURE_RANGES = {
    "deepseek": (0.0, 2.0),
    "yandex": (0.0, 1.0),
    "claude": (0.0, 1.0),
    "huggingface": (0.0, 2.0),
    "huggingface-magnum": (0.0, 2.0),
    "huggingface-tinyllama": (0.0, 2.0),
}

def _clamp_temperature(provider: str, value: float) -> float:
    min_value, max_value = _TEMPERATURE_RANGES.get(provider, (0.0, 2.0))
    return max(min_value, min(max_value, value))

def normalize_temperatures(temperatures: dict[str, float] | None) -> dict[str, float]:
    if not temperatures:
        return {}
    normalized: dict[str, float] = {}
    for provider, value in temperatures.items():
        if isinstance(value, (int, float)):
            normalized[provider] = _clamp_temperature(provider, float(value))
    return normalized

def _get_temperature(user_data: dict, provider: str, fallback: float) -> float:
    temps = user_data.get(TEMPERATURE_KEY) or {}
    value = temps.get(provider, fallback)
    if isinstance(value, (int, float)):
        return _clamp_temperature(provider, float(value))
    return _clamp_temperature(provider, float(fallback))

def _detect_language_code(text: str, fallback: str = DEFAULT_LANGUAGE_CODE) -> str:
    if not text:
        return fallback
    counts = {
        "ru": len(re.findall(r"[А-Яа-яЁё]", text)),
        "zh": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "ja": len(re.findall(r"[\u3040-\u30ff]", text)),
        "ko": len(re.findall(r"[\uac00-\ud7af]", text)),
        "ar": len(re.findall(r"[\u0600-\u06FF]", text)),
        "he": len(re.findall(r"[\u0590-\u05FF]", text)),
        "en": len(re.findall(r"[A-Za-z]", text)),
    }
    lang, score = max(counts.items(), key=lambda item: item[1])
    if score == 0:
        return fallback
    return lang

def _build_payload(
    answer: str,
    provider: str,
    processing_time_ms: int,
    usage: dict[str, int] | None = None,
    temperature: float | None = None,
) -> dict:
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%H:%M:%S - %d.%m.%Y")
    model_label = provider
    if provider == "huggingface":
        model_label = (HF_MODEL_ID or "huggingface").split("/")[-1]
    if provider == "huggingface-magnum":
        model_label = (HF_MODEL_MAGNUM_ID or "magnum").split("/")[-1]
    if provider == "huggingface-tinyllama":
        model_label = (HF_MODEL_TLAMA_ID or "tinyllama").split("/")[-1]
    return {
        "id": str(uuid.uuid4()),
        "time": timestamp,
        "temperature": temperature,
        "model": model_label,
        "language": _detect_language_code(answer),
        "processing_time_ms": processing_time_ms,
        "status": "success",
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "answer": answer,
    }


def _format_payload(
    answer: str,
    json_mode: str,
    provider: str,
    processing_time_ms: int,
    usage: dict[str, int] | None = None,
    temperature: float | None = None,
) -> str:
    if json_mode == JSON_MODE_OFF:
        return answer
    payload = _build_payload(answer, provider, processing_time_ms, usage, temperature)
    if json_mode == JSON_MODE_CLEAN:
        return json.dumps(payload, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _handle_command(text: str, user_data: dict, chat_data: dict) -> list[str] | None:
    command = text.strip().lower()
    if command == "/discussion_toggle":
        current = bool(user_data.get(DISCUSSION_MODE_KEY, False))
        user_data[DISCUSSION_MODE_KEY] = not current
        return ["Режим обсуждения включен." if not current else "Режим обсуждения выключен."]
    if command == "/discussion_on":
        user_data[DISCUSSION_MODE_KEY] = True
        return ["Режим обсуждения включен."]
    if command == "/discussion_off":
        user_data[DISCUSSION_MODE_KEY] = False
        return ["Режим обсуждения выключен."]
    if command == "/use_deepseek":
        user_data[AI_PROVIDER_KEY] = DEFAULT_PROVIDER
        return ["Модель: Deepseek"]
    if command == "/use_yandex":
        user_data[AI_PROVIDER_KEY] = "yandex"
        return ["Модель: Yandex Cloud"]
    if command == "/use_claude":
        user_data[AI_PROVIDER_KEY] = "claude"
        return ["Модель: Claude"]
    if command == "/use_huggingface":
        user_data[AI_PROVIDER_KEY] = "huggingface"
        return ["Модель: Hugging Face"]
    if command == "/use_huggingface_magnum":
        user_data[AI_PROVIDER_KEY] = "huggingface-magnum"
        return ["Модель: Hugging Face Magnum"]
    if command == "/use_huggingface_tinyllama":
        user_data[AI_PROVIDER_KEY] = "huggingface-tinyllama"
        return ["Модель: TinyLlama"]
    if command == "/json_toggle":
        current = user_data.get(JSON_MODE_KEY, JSON_MODE_PRETTY)
        if current == JSON_MODE_OFF:
            user_data[JSON_MODE_KEY] = JSON_MODE_PRETTY
            return ["JSON-режим включен."]
        user_data[JSON_MODE_KEY] = JSON_MODE_OFF
        return ["JSON-режим выключен."]
    if command == "/json_on":
        user_data[JSON_MODE_KEY] = JSON_MODE_PRETTY
        return ["JSON-режим включен."]
    if command == "/json_clean":
        user_data[JSON_MODE_KEY] = JSON_MODE_CLEAN
        return ["JSON-режим: чистый JSON без форматирования."]
    if command == "/json_off":
        user_data[JSON_MODE_KEY] = JSON_MODE_OFF
        return ["JSON-режим выключен."]
    if command == "/system_prompt":
        return [user_data.get(SYSTEM_PROMPT_KEY, SYSTEM_PROMPT)]
    if command == "/prompt_templates":
        prompt_templates = {
            "simple": SIMPLE_PROMT,
            "assistant": SYSTEM_PROMPT,
            "mathematician": MATHEMATICIAN_PROMPT,
            "philosopher": PHILOSOPHER_PROMPT,
            "creative": CREATIVE_PROMPT,
        }
        return [json.dumps(prompt_templates, ensure_ascii=False)]
    if command.startswith("/set_system_prompt"):
        prompt_text = text[len("/set_system_prompt"):].lstrip()
        if prompt_text:
            user_data[SYSTEM_PROMPT_KEY] = prompt_text
            return ["Системный промт обновлен."]
        user_data.pop(SYSTEM_PROMPT_KEY, None)
        return ["Системный промт сброшен к значению по умолчанию."]
    if command == "/drop_context":
        had_context = CLARIFY_STATE_KEY in user_data
        user_data.pop(CLARIFY_STATE_KEY, None)
        return ["Контекст уточнений сброшен." if had_context else "Контекст уточнений уже пуст."]
    if command == "/reset_chat":
        user_data.clear()
        chat_data.clear()
        return ["Чат сброшен."]
    return None


def process_text(text: str, user_data: dict, chat_data: dict) -> list[str]:
    if not text:
        return []

    command_result = _handle_command(text, user_data, chat_data)
    if command_result is not None:
        return command_result

    start_time = time.perf_counter()
    provider = user_data.get(AI_PROVIDER_KEY, DEFAULT_PROVIDER)
    json_mode = user_data.get(JSON_MODE_KEY, JSON_MODE_PRETTY)
    system_prompt = user_data.get(SYSTEM_PROMPT_KEY, SYSTEM_PROMPT)
    temperature_by_provider = user_data.get(TEMPERATURE_KEY, _DEFAULT_TEMPERATURES)

    try:
        discussion_mode = user_data.get(DISCUSSION_MODE_KEY, False)
        if discussion_mode:
            answers = generate_discussion_answers(text, temperature_by_provider)
            discussion_memory = {label: content for _, label, content, _ in answers}
            chat_data[DISCUSSION_MEMORY_KEY] = discussion_memory
            output = []
            for answer_provider, role, content, usage in answers:
                answer_temperature = _get_temperature(user_data, answer_provider, 0.6)
                output.append(
                    _format_payload(
                        f"{role}:\n{(content or '...').strip()}",
                        json_mode,
                        answer_provider,
                        int((time.perf_counter() - start_time) * 1000),
                        usage,
                        answer_temperature,
                    )
                )
            referee_text, referee_usage = generate_referee_answer(
                discussion_memory,
                _get_temperature(user_data, DEFAULT_PROVIDER, 0.6),
            )
            output.append(
                _format_payload(
                    f"REFEREE:\n{(referee_text or '...').strip()}",
                    json_mode,
                    DEFAULT_PROVIDER,
                    int((time.perf_counter() - start_time) * 1000),
                    referee_usage,
                    _get_temperature(user_data, DEFAULT_PROVIDER, 0.6),
                )
            )
            return output

        clarify_state = user_data.get(CLARIFY_STATE_KEY)
        if clarify_state:
            last_question = clarify_state.get("last_question")
            if last_question:
                clarify_state.setdefault("qas", []).append(
                    {"question": last_question, "answer": text}
                )
            question, question_usage = generate_next_question(
                clarify_state["original"],
                clarify_state.get("qas", []),
                clarify_state.get("asked", []),
                provider,
                system_prompt,
                _get_temperature(user_data, provider, 0.6),
            )
            if question:
                asked = clarify_state.setdefault("asked", [])
                normalized_question = question.strip().lower()
                if any(q.strip().lower() == normalized_question for q in asked):
                    question = None
                else:
                    asked.append(question)
            if question:
                clarify_state["last_question"] = question
                return [
                    _format_payload(
                        question,
                        json_mode,
                        provider,
                        int((time.perf_counter() - start_time) * 1000),
                        question_usage,
                        _get_temperature(user_data, provider, 0.6),
                    )
                ]

            summary, summary_usage = summarize_with_answers(
                clarify_state["original"],
                [qa["answer"] for qa in clarify_state.get("qas", [])],
                provider,
                _get_temperature(user_data, provider, 0.6),
            )
            if not summary:
                summary = "Не фортануло, смог сформировать ответ. Попробуй перефразировать."

            user_data.pop(CLARIFY_STATE_KEY, None)
            return [
                _format_payload(
                    summary,
                    json_mode,
                    provider,
                    int((time.perf_counter() - start_time) * 1000),
                    summary_usage,
                    _get_temperature(user_data, provider, 0.6),
                )
            ]

        question, question_usage = generate_next_question(
            text,
            [],
            [],
            provider,
            system_prompt,
            _get_temperature(user_data, provider, 0.6),
        )
        if question:
            user_data[CLARIFY_STATE_KEY] = {
                "original": text,
                "qas": [],
                "asked": [question],
                "last_question": question,
            }
            return [
                _format_payload(
                    question,
                    json_mode,
                    provider,
                    int((time.perf_counter() - start_time) * 1000),
                    question_usage,
                    _get_temperature(user_data, provider, 0.6),
                )
            ]

        summary, summary_usage = summarize_with_answers(
            text,
            [],
            provider,
            _get_temperature(user_data, provider, 0.6),
        )
        if not summary:
            summary = "Не фортануло, смог сформировать ответ. Попробуй перефразировать."

        return [
            _format_payload(
                summary,
                json_mode,
                provider,
                int((time.perf_counter() - start_time) * 1000),
                summary_usage,
                _get_temperature(user_data, provider, 0.6),
            )
        ]

    except Exception as exc:
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        logging.error("Ошибка в process_text (%s ms): %s", processing_time_ms, exc)
        return [
            _format_payload(
                f"Ошибка: {str(exc)[:200]}",
                json_mode,
                provider,
                int((time.perf_counter() - start_time) * 1000),
                _get_temperature(user_data, provider, 0.6),
            )
        ]

