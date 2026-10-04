"""Конфигурация Telegram-бота."""

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str
    backend_url: str = "http://127.0.0.1:8000"
    bot_admin_ids: list[int] = []
    admin_token: SecretStr = SecretStr("")
    broadcast_poll_interval_seconds: float = 10.0

    # --- новые поля ---
    internal_token: SecretStr
    bot_api_port: int = 9000

    @field_validator("bot_admin_ids", mode="before")
    @classmethod
    def _parse_admin_ids(cls, v):
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                v = v[1:-1]
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v


bot_config = BotConfig()
