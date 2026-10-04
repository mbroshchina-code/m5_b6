"""Модуль системной диагностики состояния приложения (Health Check API).

"""

from fastapi import APIRouter, Request, Response

# Инициализируем роутер с тегом для Swagger UI
router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    """Моментальная проверка жизнеспособности сервера (Liveness Probe).
    
    Всегда возвращает статус 200 OK без проверки сетевых зависимостей.
    """
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request, response: Response) -> dict:
    """Глубокая проверка готовности инфраструктуры к запросам (Readiness Probe).
    
    Выполняет асинхронный ping до Redis. В случае успеха возвращает HTTP 200.
    В случае сбоя или таймаута — возвращает HTTP 503 строго по ТЗ Задачи №4.
    """
    cache = getattr(request.app.state, "redis", None)
       
    # Если клиент Redis инициализирован в lifespan
    if cache is not None:
        try:
            # Делаем асинхронную проверку связи с Redis
            await cache.ping()
            
            # ТРЕБОВАНИЕ ЗАДАЧИ 4: Успешный ответ (HTTP 200 выставляется по умолчанию)
            return {
                "status": "ok", 
                "redis": "up"
            }
        except Exception:
            # Сюда мы попадаем, если ping упал с ошибкой соединения или таймаутом
            pass

    # ТРЕБОВАНИЕ ЗАДАЧИ 4: Если Redis недоступен, принудительно возвращаем статус 503
    response.status_code = 503
    return {
        "status": "degraded", 
        "redis": "down"
    }