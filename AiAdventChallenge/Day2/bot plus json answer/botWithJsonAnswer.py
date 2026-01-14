import asyncio
import json
import html
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from telegram import Update
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


async def send_json_message(update: Update, payload: dict) -> None:
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_text = html.escape(json_text)
    await update.message.reply_text(
        f"<pre><code class=\"language-json\">{json_text}</code></pre>",
        parse_mode="HTML",
    )


# =========================
# Handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%H:%M:%S - %d.%m.%Y")
    payload = {
        "id": str(uuid.uuid4()),
        "time": timestamp,
        "model": MODEL,
        "language": "ru",
        "processing_time_ms": 0,
        "status": "success",
        "answer": "Йо, йо, ой! Как сам? 🙂",
    }
    await send_json_message(update, payload)


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


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text
    chat_type = update.effective_chat.type
    bot_username = context.bot.username

    # В группах — только по упоминанию
    if chat_type in ("group", "supergroup"):
        mention = f"@{bot_username}"
        if mention not in text:
            return
        text = text.replace(mention, "").strip()

    language = getattr(update.effective_user, "language_code", None) or "und"
    start_time = time.perf_counter()
    
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%H.%M.%S - %d.%m.%Y")
    payload = {
        "id": str(uuid.uuid4()),
        "time": timestamp,
        "model": MODEL,
        "language": language[:2] if len(language) >= 2 else "und",
        "processing_time_ms": 0,
        "status": "error",
        "answer": "",
    }

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Отвечай кратко и по делу, со сленговыми фразами. "
                        "Отвечай ТОЛЬКО текстом ответа, без каких-либо пояснений, "
                        "форматирования или дополнительных комментариев. "
                        "Просто дай чистый текстовый ответ на вопрос пользователя."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.7,
        )

        raw_answer = response.choices[0].message.content.strip()
        
        # Очищаем ответ от возможного JSON/markdown
        cleaned_answer = raw_answer 
        
        # Если ответ пустой после очистки
        if not cleaned_answer:
            cleaned_answer = "Не фортануло, смог сформировать ответ. Попробуй перефразировать."

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        payload.update({
            "time": datetime.now(timezone(timedelta(hours=3))).strftime("%H.%M.%S - %d.%m.%Y"),
            "processing_time_ms": processing_time_ms,
            "status": "success",
            "answer": cleaned_answer[:4000],
        })

    except Exception as e:
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        payload.update({
            "time": datetime.now(timezone(timedelta(hours=3))).strftime("%H.%M.%S - %d.%m.%Y"),
            "processing_time_ms": processing_time_ms,
            "status": "error",
            "answer": f"Ошибка: {str(e)[:200]}",
        })
        logging.error(f"Ошибка в on_text: {e}")

    # Отправляем ответ в формате JSON
    await send_json_message(update, payload)


# =========================
# Main
# =========================
def main():
    logging.info("🤖 Инициализация Telegram Application")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logging.info("📡 Запуск polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
