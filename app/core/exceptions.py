"""Модуль кастомных доменных исключений приложения.

Используется для перехвата сырых ошибок OpenAI SDK и их маппинга в HTTP-статусы.
"""

class LLMError(Exception):
    """Базовое доменное исключение слоя LLM."""
    def __init__(self, message: str = "Внутренняя ошибка провайдера ИИ"):
        self.message = message
        super().__init__(self.message)


class LLMRateLimitError(LLMError):
    """Провайдер вернул rate limit (HTTP 429)."""
    def __init__(self, message: str = "Превышен лимит запросов к ИИ. Попробуйте позже."):
        super().__init__(message)


class LLMAuthError(LLMError):
    """Невалидный ключ или 401/403 от провайдера (HTTP 502)."""
    def __init__(self, message: str = "Ошибка аутентификации на стороне провайдера ИИ."):
        super().__init__(message)


class LLMTimeoutError(LLMError):
    """LLM не ответил вовремя (HTTP 504)."""
    def __init__(self, message: str = "Превышено время ожидания ответа от сервера ИИ."):
        super().__init__(message)


class LLMContentFilterError(LLMError):
    """Контент заблокирован модерацией (HTTP 400)."""
    def __init__(self, message: str = "Запрос заблокирован системой фильтрации содержимого."):
        super().__init__(message)
