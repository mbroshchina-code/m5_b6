"""Модуль маскирования персональных данных (PII) и хэширования промптов.

Адаптирован на основе эталонного референса регулярных выражений наставника.
"""

import hashlib
import re

# ТСтрогий порядок и оптимальные регулярные выражения
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email
    (re.compile(r"[\w.\-]+@[\w.\-]+\.\w+"), "[EMAIL]"),
    # Карта: ровно 16 цифр, возможно с пробелами
    (re.compile(r"\b(?:\d{4}[ \-]?){3}\d{4}\b"), "[CARD]"),
    # Телефон: строгий формат РФ/международный
    (
        re.compile(
            r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
            r"|\+?\d{1,3}[\s\-]\d{1,4}[\s\-]\d{1,4}[\s\-]\d{1,4}"
        ),
        "[PHONE]",
    ),
]


def redact_pii(text: str) -> str:
    """Маскирует чувствительные данные (email, карты, телефоны) с помощью регулярных выражений."""
    if not text:
        return text
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def prompt_hash(text: str) -> str:
    """Генерирует SHA-256 хэш от сырой строки промпта для вывода в структурированный JSON-лог."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
