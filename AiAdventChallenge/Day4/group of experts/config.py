import logging
import os
from pathlib import Path


# =========================
# Загрузка токенов
# =========================

def load_tokens(path="tokens.txt") -> dict:
    tokens = {}
    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(
            f"❌ Файл {path} не найден.\n"
            "Создайте файл и добавьте:\n"
            "DEEPSEEK_API_KEY=..."
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

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


configure_logging()
logging.info("🚀 Сервис запускается...")


# =========================
# Токены
# =========================

tokens = load_tokens()

DEEPSEEK_API_KEY = tokens.get("DEEPSEEK_API_KEY")
YANDEX_CLOUD_API_KEY = (
    tokens.get("YANDEX_CLOUD_API_KEY")
    or os.getenv("YANDEX_CLOUD_API_KEY")
)
YANDEX_PROJECT_ID = (
    tokens.get("YANDEX_PROJECT_ID")
    or tokens.get("YANDEX_PROJECT")
    or os.getenv("YANDEX_PROJECT_ID")
    or os.getenv("YANDEX_PROJECT")
)
YANDEX_PROMPT_ID = (
    tokens.get("YANDEX_PROMPT_ID")
    or os.getenv("YANDEX_PROMPT_ID")
)
YANDEX_MODEL_ID = (
    tokens.get("YANDEX_MODEL_ID")
    or os.getenv("YANDEX_MODEL_ID")
)
CLAUDE_API_KEY = (
    tokens.get("CLAUDE_API_KEY")
    or tokens.get("CLAUD_API_KEY")
    or tokens.get("ANTHROPIC_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("CLAUDE_API_KEY")
)
CLAUDE_MODEL = tokens.get("CLAUDE_MODEL_ID") or os.getenv("CLAUDE_MODEL") or "claude-3-haiku-20240307"
CLAUDE_BASE_URL = tokens.get("CLAUDE_BASE_URL") or "https://api.anthropic.com"

if not DEEPSEEK_API_KEY:
    raise RuntimeError("❌ DEEPSEEK_API_KEY не найден в tokens.txt")


def _validate_yandex_credentials() -> None:
    provided_any = any(
        (
            YANDEX_CLOUD_API_KEY,
            YANDEX_PROJECT_ID,
            YANDEX_PROMPT_ID,
            YANDEX_MODEL_ID,
        )
    )
    if not provided_any:
        return

    missing = []
    if not YANDEX_CLOUD_API_KEY:
        missing.append("YANDEX_CLOUD_API_KEY")
    if not YANDEX_PROJECT_ID:
        missing.append("YANDEX_PROJECT_ID/YANDEX_PROJECT")
    if missing:
        raise RuntimeError(
            "❌ Недостаточно данных для Yandex Cloud (REST Assistant API): "
            f"укажите {', '.join(missing)} в tokens.txt или удалите все YANDEX_* строки."
        )
    if not (YANDEX_PROMPT_ID or YANDEX_MODEL_ID):
        raise RuntimeError(
            "❌ Для Yandex Cloud (REST Assistant API) нужно указать "
            "YANDEX_PROMPT_ID или YANDEX_MODEL_ID в tokens.txt."
        )


_validate_yandex_credentials()
