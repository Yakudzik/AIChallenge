import asyncio
import json
import html
import logging
import time
import uuid
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from telegram import Update, BotCommand, MenuButtonCommands
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI


# =========================
# Загрузка токенов
# =========================
def load_tokens(path="tokens.txt") -> dict:
    tokens = {}
    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(
            f"❌ Файл {path} не найден.\n"
            f"Создайте файл и добавьте:\n"
            f"TELEGRAM_BOT_TOKEN=...\n"
            f"DEEPSEEK_API_KEY=..."
        )

    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(f"Неверный формат строки в {path}: {line}")

        key, value = line.split("=", 1)
        tokens[key.strip()] = value.strip()

    return tokens


# =========================
# Логи
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("🚀 Бот запускается...")


# =========================
# Токены
# =========================
tokens = load_tokens()

TELEGRAM_BOT_TOKEN = tokens.get("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = tokens.get("DEEPSEEK_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN не найден в tokens.txt")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("❌ DEEPSEEK_API_KEY не найден в tokens.txt")


# =========================
# AI клиент
# =========================
MODEL = "deepseek-chat"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


CLARIFY_STATE_KEY = "clarify_state"


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


def build_payload(language: str, status: str, answer: str, processing_time_ms: int) -> dict:
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%H:%M:%S - %d.%m.%Y")
    return {
        "id": str(uuid.uuid4()),
        "time": timestamp,
        "model": MODEL,
        "language": language,
        "processing_time_ms": processing_time_ms,
        "status": status,
        "answer": answer,
    }


async def send_json_message(update: Update, payload: dict) -> None:
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_text = html.escape(json_text)
    await update.message.reply_text(
        f"<pre><code class=\"language-json\">{json_text}</code></pre>",
        parse_mode="HTML",
    )


def generate_next_question(
    original: str,
    qas: list[dict[str, str]],
    asked: list[str],
) -> str | None:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (

"Ты — диалоговая LLM-модель-ассистент. Ты ведёшь один непрерывный диалог и сохраняешь контекст между сообщениями."
"Ты используешь внутренний протокол: CLARIFY → ANSWER → SUMMARY, но НИКОГДА не показываешь пользователю названия состояний или внутренние метки."
"Пользователь никогда не должен видеть слова: 'CLARIFY', 'ANSWER', 'SUMMARY', 'state', 'состояние'."
""
"=== ГЛАВНАЯ ЦЕЛЬ ==="
"Быстро получить недостающие параметры (если это критично), затем выполнить задачу и выдать результат + суммари цепочки."
""
"=== ПРАВИЛО КОНТЕКСТА (ANTI-RESET) ==="
"Запрещено 'сбрасывать' диалог: ты всегда продолжаешь с учётом предыдущих сообщений."
"Если пользователь уже задал задачу, ты НЕ возвращаешься к стартовому 'Чем помочь?' и НЕ ведёшь себя как в новом чате."
""
"=== ПРАВИЛО ПРИВЕТСТВИЙ (NO-MULTI-HELLO) ==="
"Ты можешь поздороваться ТОЛЬКО в первом сообщении ассистента в рамках диалога."
"После первого сообщения ассистента слово 'Привет' и любые приветствия запрещены, даже если пользователь проявляет эмоции."
"Исключение: если пользователь снова явно пишет приветствие ('привет', 'hello', 'здравствуй'), можно ответить коротким приветствием ОДИН раз и сразу продолжить текущую задачу без сброса контекста."
""
"=== ПРАВИЛО НЕ ГОВОРИТЬ ОТ ЛИЦА ПОЛЬЗОВАТЕЛЯ ==="
"Запрещено перефразировать запрос пользователя как будто это твои желания (например: 'Отлично, хочу приготовить пирог')."
"Можно кратко подтвердить нейтрально без шаблона 'Понял:' и без двоеточия."
"Примеры: 'Хорошо, принял.' или 'Ок, понял, продолжаю.'"
""
"=== ПРАВИЛО КОРОТКОГО ПОДТВЕРЖДЕНИЯ ==="
"Когда пользователь даёт параметр (например: 'сладкий'), ответь одной короткой фразой фиксации (до 10 слов), затем продолжай вопросами/решением."
"Не используй шаблон 'Понял:' и двоеточие."
"Пример: 'Хорошо, сладкий пирог.'"
""
"=== ПРАВИЛО НЕПУСТОГО ОТВЕТА ==="
"Каждое сообщение ассистента должно содержать либо:"
"(A) конкретные вопросы (1–5) с '?', либо"
"(B) готовый результат/решение, либо"
"(C) короткую просьбу дать входные данные + 1–3 вопроса с '?'."
"Запрещены ответы из одного слова/метки/заголовка."
""
"=== КОНТРАКТ УТОЧНЕНИЙ (ANTI-LOOP) ==="
"Если ты решаешь уточнять, твой ответ ОБЯЗАН содержать:"
"1) РОВНО 1 короткую фразу (до 20 слов), чего не хватает."
"2) Далее список 1–5 конкретных вопросов, и каждый вопрос обязан содержать '?'."
"Если в ответе про уточнение нет ни одного '?', это ОШИБКА: перепиши ответ и добавь вопросы."
"Запрещено отвечать общими фразами ('не хватает информации', 'нужно уточнить параметры', 'зависит от контекста') без вопросов."
""
"=== ЛИМИТ УТОЧНЕНИЙ ==="
"Максимум 2 раунда уточнений на одну задачу."
"После 2 раундов: сделай разумные 'Допущения' по оставшимся параметрам и выполни задачу."
""
"=== КАК ЗАДАВАТЬ ВОПРОСЫ ==="
"Задавай вопросы от самых важных к менее важным."
"По возможности предлагай варианты ответов (A/B/C) или примеры."
"Если пользователь спрашивает 'Какая информация нужна?' — сразу выдай список конкретных вопросов (1–5) с '?', без общих объяснений."
"Если пользователь отвечает 'ок/да/не знаю' или без параметров — повтори те же вопросы короче и с вариантами."
""
"=== ВНУТРЕННЯЯ ПАМЯТЬ (НЕ ПОКАЗЫВАЙ) ==="
"Веди внутреннюю структуру и обновляй её после каждого ответа пользователя:"
"- Goal"
"- Context"
"- Constraints"
"- Assumptions"
"- Open questions"
""
"=== КОГДА ВЫПОЛНЯТЬ ЗАДАЧУ ==="
"Если данных достаточно — выполняй задачу сразу."
"Если данных не хватает, но можно сделать качественный ответ с допущениями — делай допущения и выполняй."
""
"=== ФОРМАТ ВЫПОЛНЕНИЯ ==="
"Когда ты выдаёшь результат, НЕ добавляй служебных блоков, summary или меток."
"Выводи только ответ по задаче для пользователя."
""
"=== АБСТРАКТНАЯ ЛОГИКА УТОЧНЕНИЙ ==="
"Шаг 1: Определи 1–5 критических неизвестных параметров, без которых ответ будет бесполезен."
"Шаг 2: Задай 1–5 вопросов, закрывающих эти параметры (каждый с '?')."
"Шаг 3: После 2 раундов уточнений сделай допущения и перейди к решению."



                ),
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
        temperature=0.6,
    )
    raw = response.choices[0].message.content.strip()
    if not raw:
        return None
    normalized = normalize_lines(raw)
    if not normalized:
        normalized = [raw.strip()]
    combined = "\n".join(normalized).strip()
    if combined.lower() in {"нет", "не нужно", "достаточно", "без вопросов"}:
        return None
    return combined


def summarize_with_answers(original: str, answers: list[str]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Собери один краткий итоговый ответ на русском, "
                    "объедини исходный запрос и уточнения пользователя. "
                    "Запрещены приветствия и обращения от лица пользователя. "
                    "Выводи только итоговый текст."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Исходный запрос: {original}\n"
                    f"Уточнения: {json.dumps(answers, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()


# =========================
# Handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await on_text(update, context, text_override="Привет")


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    me = await context.bot.get_me()
    response_text = (
        "🏓 Pong!\n"
        f"Бот: @{me.username}\n"
        f"ID чата: {update.effective_chat.id}\n"
        f"Тип чата: {update.effective_chat.type}\n"
        "Статус: работает ✅"
    )
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%H:%M:%S - %d.%m.%Y")
    payload = {
        "id": str(uuid.uuid4()),
        "time": timestamp,
        "model": MODEL,
        "language": "ru",
        "processing_time_ms": 0,
        "status": "success",
        "answer": response_text,
    }
    await send_json_message(update, payload)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text_override: str | None = None):
    if not update.message or not update.message.text:
        return

    text = text_override or update.message.text
    chat_type = update.effective_chat.type
    bot_username = context.bot.username

    # В группах — только по упоминанию
    if chat_type in ("group", "supergroup"):
        mention = f"@{bot_username}"
        if mention not in text:
            return
        text = text.replace(mention, "").strip()

    language = getattr(update.effective_user, "language_code", None) or "und"
    language_code = language[:2] if len(language) >= 2 else "und"
    start_time = time.perf_counter()

    try:
        clarify_state = context.user_data.get(CLARIFY_STATE_KEY)

        if clarify_state:
            last_question = clarify_state.get("last_question")
            if last_question:
                clarify_state.setdefault("qas", []).append(
                    {"question": last_question, "answer": text}
                )
            question = await asyncio.to_thread(
                generate_next_question,
                clarify_state["original"],
                clarify_state.get("qas", []),
                clarify_state.get("asked", []),
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
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                payload = build_payload(language_code, "success", question[:4000], processing_time_ms)
                await send_json_message(update, payload)
                return

            summary = await asyncio.to_thread(
                summarize_with_answers,
                clarify_state["original"],
                [qa["answer"] for qa in clarify_state.get("qas", [])],
            )
            if not summary:
                summary = "Не фортануло, смог сформировать ответ. Попробуй перефразировать."

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            payload = build_payload(language_code, "success", summary[:4000], processing_time_ms)
            context.user_data.pop(CLARIFY_STATE_KEY, None)
            await send_json_message(update, payload)
            return

        question = await asyncio.to_thread(generate_next_question, text, [], [])
        if question:
            context.user_data[CLARIFY_STATE_KEY] = {
                "original": text,
                "qas": [],
                "asked": [question],
                "last_question": question,
            }
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            payload = build_payload(language_code, "success", question[:4000], processing_time_ms)
            await send_json_message(update, payload)
            return

        summary = await asyncio.to_thread(summarize_with_answers, text, [])
        if not summary:
            summary = "Не фортануло, смог сформировать ответ. Попробуй перефразировать."

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        payload = build_payload(language_code, "success", summary[:4000], processing_time_ms)
        await send_json_message(update, payload)
        return

    except Exception as e:
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        payload = build_payload(
            language_code,
            "error",
            f"Ошибка: {str(e)[:200]}",
            processing_time_ms,
        )
        logging.error(f"Ошибка в on_text: {e}")

    # Отправляем ответ в формате JSON
    await send_json_message(update, payload)


# =========================
# Main
# =========================
async def setup_bot(app):
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Запуск"),
            BotCommand("ping", "Проверка связи"),
        ]
    )
    await app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())


def main():
    logging.info("🤖 Инициализация Telegram Application")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.post_init = setup_bot

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logging.info("📡 Запуск polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
