"""Проверка простого admin-токена из заголовка."""

import hmac
from typing import Annotated

from fastapi import Header, HTTPException

from app.core.config import get_settings


async def require_admin(
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    expected = get_settings().admin_token.get_secret_value()
    if not expected or not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="invalid admin token")
