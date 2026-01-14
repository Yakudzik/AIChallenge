import asyncio
import json
import logging
import uuid
from datetime import datetime
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


# =========================
# Handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Йо, биатч! Как сам? 🙂")


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    me = await context.bot.get_me()
    await update.message.reply_text(
        "🏓 Pong!\n"
        f"Бот: @{me.username}\n"
        f"ID чата: {update.effective_chat.id}\n"
        f"Тип чата: {update.effective_chat.type}\n"
        "Статус: работает ✅"
    )


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

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты умный, но говоришь на сленге. "
                        "Отвечай кратко и по делу, со сленговыми фразами. "
                        "Никогда не добавляй пояснения вне JSON. "
                        "Ответ должен быть JSON-объектом с полями "
                        "`id`, `answer`, `title` и `time`, например "
                        "{\"id\": \"1\", \"answer\": \"ответ модели\", "
                        "\"title\": \"Заголовок\", \"time\": \"2025-01-01T12:00:00Z\"}."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )

        raw_answer = response.choices[0].message.content.strip()
        parsed_answer = None

        try:
            payload = json.loads(raw_answer)
            if isinstance(payload, dict):
                for key in ("answer", "response", "message", "text"):
                    if key in payload and payload[key]:
                        parsed_answer = payload[key]
                        break
        except json.JSONDecodeError:
            pass

        final_answer = str(parsed_answer or raw_answer).strip()
        if not final_answer:
            final_answer = "Фак, не смог сформировать ответ. Попробуй перефразировать."

        title = text.strip()
        if not title:
            title = "Без заголовка"
        elif len(title) > 64:
            title = title[:61] + "..."

        payload = {
            "id": str(uuid.uuid4()),
            "answer": final_answer[:4000],
            "title": title,
            "time": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }

        await update.message.reply_text(json.dumps(payload, ensure_ascii=False))

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}"[:4096])


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
