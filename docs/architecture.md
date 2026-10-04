# Системная архитектура проекта Bag Assistant (Архитектурный паспорт + ADR)

Данный документ описывает целевую архитектуру проекта, архитектурные решения (ADR), потоки данных, стратегии отказоустойчивости и текущий статус реализации каждого компонента.

## 1. Диаграмма компонентов (целевая архитектура)

Схема разделена на 4 изолированных слоя. Реализованные компоненты помечены зеленым, компоненты в плане — серым пунктиром.

```mermaid
graph TD
    classDef gatewayStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef serviceStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef llmStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef dataStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef extStyle fill:#eceff1,stroke:#455a64,stroke-width:2px,stroke-dasharray: 5 5;
    classDef plannedStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph Layer_1_Gateway [1. Layer: API Gateway]
        direction TB
        NGINX[Reverse Proxy: Nginx]:::plannedStyle
        AUTH[Auth: Middleware]:::plannedStyle
        RL[Rate Limiter: enforce_rate_limit]:::plannedStyle
        Router[FastAPI Router: routes.py]:::gatewayStyle
        Guard[Streaming Guard Middleware]:::gatewayStyle
        NGINX -.->|planned| AUTH
        AUTH -.->|planned| RL
        RL -.->|planned| Router
        Router --> Guard
    end

    subgraph Layer_2_Service [2. Layer: Application Service]
        direction TB
        ChatServ[ChatService: service.py]:::serviceStyle
        LLMServ[LLMService: llm.py]:::serviceStyle
        Expand[Query Expansion]:::serviceStyle
        Search[Local Bug Search]:::serviceStyle
        TokenCalc[Token Budget: tiktoken o200k_base]:::serviceStyle
        ModServ[ModerationService]:::serviceStyle
        ChatServ --> Expand
        Expand --> Search
        Search --> ChatServ
        ChatServ --> TokenCalc
        TokenCalc --> LLMServ
        ChatServ --> ModServ
    end

    subgraph Layer_3_LLM [3. Layer: LLM Orchestration & Cache]
        direction TB
        CacheAside{Cache-Aside: Redis<br/>TTL: 1h}:::llmStyle
        LiteLLM[LiteLLM Proxy<br/>Circuit Breaker]:::plannedStyle
        OpenAI[Primary: OpenAI GPT-4o-mini]:::llmStyle
        OpenRouter[Secondary: OpenRouter GPT-OSS<br/>Free Tier]:::plannedStyle
        Ollama[Tertiary: Ollama Qwen 2.5<br/>Local]:::plannedStyle
        LLMServ -->|1. Check Cache| CacheAside
        CacheAside -->|Cache Hit| LLMServ
        CacheAside -->|Cache Miss| LLMServ
        LLMServ -.->|planned: via LiteLLM| LiteLLM
        LiteLLM -.->|2. Try Primary| OpenAI
        LiteLLM -.->|3. Fallback| OpenRouter
        LiteLLM -.->|4. Offline| Ollama
    end

    subgraph Layer_4_Data [4. Layer: Data & Infrastructure]
        direction TB
        RepoFactory[Repository Factory: deps.py]:::dataStyle
        JsonRepo[JsonChatRepository: json_repo.py<br/>ACTIVE]:::dataStyle
        PgRepo[PostgresChatRepository: pg_repo.py<br/>NOT CONNECTED]:::dataStyle
        BugsDB[(bugs_database.json)]:::dataStyle
    end

    Ext_Client[Client: Telegram Bot / CLI / Web]
    Ext_Redis[Redis Stack]:::extStyle
    Ext_Postgres[PostgreSQL DB]:::extStyle

    Ext_Client -->|HTTP Request| NGINX
    Ext_Client -->|direct access| Router
    Guard -->|Forward| ChatServ
    LLMServ -->|Cache Read/Write| Ext_Redis
    ChatServ -->|Load History| RepoFactory
    RepoFactory -->|CHAT_REPOSITORY = json| JsonRepo
    RepoFactory -.->|CHAT_REPOSITORY = postgres<br/>(not active)| PgRepo
    JsonRepo -->|Append-Only| LocalDisk[(messages.jsonl)]
    PgRepo -.->|SQL Transactions| Ext_Postgres
    Search -->|Read| BugsDB

    linkStyle 4 stroke:#ff1744,stroke-width:2px;
    linkStyle 7 stroke:#2979ff,stroke-width:2px;
    linkStyle 12 stroke:#ff1744,stroke-width:2px;
    linkStyle 15 stroke:#2979ff,stroke-width:2px;
```

**Легенда статуса:**
- Зеленая заливка (`serviceStyle`, `llmStyle`) — реализовано и работает.
- Серая пунктирная заливка (`plannedStyle`) — описано в архитектуре, но **не реализовано** (требует доработки/инфраструктуры).

---

## 2. ADR 1: Выбор паттерна диалогового взаимодействия

### Context

В рамках проекта `bag_assistant` реализуется сценарий интерактивного чат-бота техподдержки и RAG-поиска по внутренней базе знаний IT-онбординга.

**Целевые метрики нагрузки и бизнес-ограничения:**
- **Интенсивность (RPM):** Средняя нагрузка составляет 50–100 запросов в минуту.
- **Объем токенов (TPM):** Ожидается пиковый расход до 50,000 TPM.
- **Размер ответа:** Средний объем генерации ассистента — 400–600 токенов OpenAI на один запрос.
- **Финансовый бюджет:** Жесткий лимит составляет **$5.00 в день**.
- **Эффективность кэширования:** Целевой показатель **Cache Hit Rate равен 40%** (четыре из десяти повторных вопросов по багам должны закрываться из кэша Redis, не тратя деньги на OpenAI).

### Decision

Выбран паттерн **Streaming (асинхронный стриминг токенов через Server-Sent Events — SSE)**.

**Обоснование выбора:**
1. **Time-to-First-Token (TTFT):** Скорость восприятия ответа пользователем критически важна для UX. Стриминг выводит первые буквы ответа на экран уже через 200–400 мс, убирая долгое ожидание полной генерации.
2. **Эффективное удержание соединений:** SSE является нативным, легковесным протоколом поверх стандартного HTTP/1.1 и HTTP/2, который не требует сложного оверхеда двусторонних сокетов (WebSockets).

### Consequences

- **Что выиграно:** Радикальное улучшение UX, экономия ресурсов серверов.
- **Что усложнилось:** Потребовалось полностью отключить буферизацию на уровне инфраструктурного прокси-сервера (добавлены заголовки `X-Accel-Buffering: no` для Nginx). Потребовалось кеширование входящего бинарного потока запроса (`await request.body()`) внутри FastAPI во избежание багов Starlette Middleware.

**Статус:** Реализовано. Guard-middleware работает. Nginx еще не поднят.

### Alternatives considered

- **Паттерн Request-Response (Отвергнут):** Вызывал задержку в 10-15 секунд для конечного пользователя на длинных ответах, что приводило к ощущению «зависания» интерфейса.
- **Асинхронная очередь задач / Queue (Отвергнут):** Паттерн с Celery/RabbitMQ и поллингом избыточен для текстового чат-бота, так как усложняет архитектуру введением воркеров.

---

## 3. ADR 2: Выбор стратегии Fault Tolerance и каскада LLM-провайдеров

### Context

Слой интеграции с языковыми моделями подвержен рискам сетевой недоступности (503), исчерпания лимитов (429) и географических блокировок (403). Сервис обязан сохранять отказоустойчивость в рамках выделенного бюджета.

### Decision

Внедрение паттерна **Circuit Breaker** через **LiteLLM Proxy** со следующей обоснованной цепочкой провайдеров (Fallback Chain):

1. **Primary: OpenAI GPT-4o-mini**
   - *Системный идентификатор:* `gpt-4o-mini`
   - *Обоснование:* Обладает наилучшим соотношением цены и качества для простых задач техподдержки. Стоимость $0.15 за миллион токенов позволяет гарантированно уложиться в лимит бюджета $5/день при нагрузке в 50,000 TPM.

2. **Secondary Fallback: OpenRouter Free Tier (GPT-OSS)**
   - *Системный идентификатор:* `openrouter/openai/gpt-oss-20b:free`
   - *Обоснование:* Активируется автоматически по API-ключу OpenRouter при блокировках, падении или исчерпании лимитов основного эндпоинта OpenAI. Использование бесплатных (Free Tier) моделей API позволяет удерживать SLA без дополнительных затрат.

3. **Tertiary Fallback: Ollama Local (Qwen 2.5)**
   - *Системный идентификатор:* `ollama/qwen2.5:7b-instruct-q4_K_M`
   - *Обоснование:* Полностью бесплатная локальная модель из семейства Qwen 2.5, развернутая внутри закрытого контура компании (порт 11434). Гарантирует базовое выживание FAQ-бота и обработку критических сценариев при полном отсутствии внешней сети или падении всех API-шлюзов.

### Consequences

- **Что выиграно:** Бесперебойная работа ассистента 24/7.
- **Что усложнилось:** Потребовалась унификация контекста истории сообщений под универсальный ChatML-стандарт для работы с разными API-провайдерами, а также поддержка локальных мощностей для инференса Qwen.

**Статус:** LiteLLM Proxy **не внедрён**. Сейчас прямой вызов OpenAI API. Fallback chain **не работает**.

---

## 4. Поток обработки запроса (Query-to-Response)

При получении сообщения от пользователя сервис проходит следующие этапы:

1. **Query Expansion** — LLM генерирует 3 синонима к запросу пользователя для расширения семантического охвата.
2. **Локальный поиск по базе багов** — расширенные запросы ищутся в `bugs_database.json`. Найденные баги ранжируются по релевантности.
3. **Fake Tool-Calling** — результаты поиска вставляются в контекст диалога как имитация вызова инструмента: сообщение `assistant` с `tool_calls` и сообщение `tool` с результатами.
4. **Классификация** — LLM получает системный промпт со шкалой релевантности (3–5 баллов — релевантные, 2 балла — менее релевантные) и классифицирует найденные баги.
5. **Streaming** — ответ возвращается клиенту через SSE (Server-Sent Events) чанками.

**В целевой архитектуре** (через LiteLLM Proxy) шаг 4 проходит через единую точку входа, которая автоматически маршрутизирует запрос на доступного провайдера. Сейчас — прямой вызов OpenAI.

---

## 5. Паттерн Cache-Aside

Для минимизации затрат на повторные однотипные запросы реализован опциональный узел Cache-Aside на базе Redis.

**Сборка ключа кэша:** детерминированный MD5-хэш от `model + json.dumps(messages, sort_keys=True) + str(temperature)`.

**Условие кэширования:** кэш активен только при `temperature=0`. При ненулевой температуре кэширование отключается, так как ответы должны быть недетерминированными.

**TTL:** 3600 секунд (1 час), переменная `CACHE_TTL_SECONDS` в `.env`.

**Статус:** Реализовано. Redis опционален, работает при наличии `REDIS_URL`.

---

## 6. Потенциальные точки отказа и Graceful Degradation

| Слой архитектуры | Точка отказа | Что произойдет при выпадении | Паттерн смягчения удара (Mitigation) | Graceful Degradation | Статус |
|:---|:---|:---|:---|:---|:---|
| **1. Gateway** | Падение инстанса Nginx / Redis Лимитера | Клиенты не могут достучаться до API бэкенда, либо rate-limit перестает работать. | **DNS Round-Robin & Fallback-кэш** | Если лежит Redis, лимитер переходит в режим `Fail-Open`: временно пропускает все запросы без ограничений по RPM, защищая доступ пользователей ценой риска повышенной нагрузки. Без 500-й страницы. | **Не реализовано.** Nginx не поднят. Rate Limiter не написан. |
| **2. Service** | Ошибка асинхронного стриминга (Обрыв сети у клиента) | Поток чанков прерывается на середине генерации текста ассистента. | **Graceful Stream Interruption** | Блок `except Exception` внутри `send_message` перехватывает обрыв, извлекает накопленный в памяти `buffer` текста и принудительно сохраняет его в файлы на диск как финальный кусок, отправляя маркер закрытия сессии. Данные не теряются. Без 500-й ошибки. | **Реализовано.** Guard + обработка обрыва работают. |
| **3. LLM Layer** | Полная недоступность внешних ИИ-провайдеров (OpenAI & OpenRouter) | API-ключи заблокированы, внешние серверы лежат. | **Каскадный Fallback LiteLLM & Шаблонные ответы** | Система размыкает Circuit Breaker и переключается на **Template-ответы из локальной базы FAQ**. Вместо генеративного ответа ИИ выводит статичную заглушку: «Сервис генерации временно недоступен. Пожалуйста, воспользуйтесь поиском по ключевым словам...». Без 500-й страницы. | **Не реализовано.** LiteLLM не поднят. Fallback chain не работает. |
| **4. Data Layer** | Сбой базы данных PostgreSQL | Репозиторий `PostgresChatRepository` не может записать историю или залогировать фидбек. | **Repository Factory Fallback (Failover к файлам)** | Фабрика зависимостей в `deps.py` ловит ошибку подключения к БД и мгновенно переключает весь чат-модуль на локальный файловый режим `JsonChatRepository`. История пишется append-only строками в `messages.jsonl` на диск, пользователь не замечает сбоя. Без 500-й ошибки. | **Частично реализовано.** Фабрика переключается на JSONL, но логика авто-failover требует доработки. |

---

## 7. LiteLLM как LLM Gateway

### Обоснование выбора: «Брать готовый LiteLLM» против «Писать роутер самим»

Вместо написания собственного велосипеда для каскадного переключения моделей (Fallback chain) на чистом Python, в проект планируется внедрение официального **LiteLLM Proxy Server**.

**Почему взят готовый LiteLLM:**
1. **Встроенный Circuit Breaker:** Сервер из коробки умеет отслеживать коды ошибок (403, 429, 5xx) и автоматически «размыкать цепь», перенаправляя трафик на резервного провайдера без перезапуска основного FastAPI приложения.
2. **Стандартизация OpenAI:** LiteLLM предоставляет единый wire-совместимый интерфейс OpenAI Completions. Наш `ChatService` отправляет запросы в единую точку, не зная, какая модель (облачная или локальная Ollama) отвечает за генерацию в данный момент. Это радикально упрощает кодовую базу.

### Конфигурация `docs/litellm/config.yaml` (целевая)

Для работы прокси-слоя подготовлен следующий конфигурационный файл:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: "os.environ/OPENAI_API_KEY"
      tpm: 50000
      rpm: 100

  - model_name: claude-haiku
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: "os.environ/ANTHROPIC_API_KEY"

  - model_name: local-fallback
    litellm_params:
      model: ollama/llama3
      api_base: "http://localhost:11434"

router_settings:
  routing_strategy: failover
  set_verbose: true
  allowed_fails: 3
  cooldown_time: 30
```

### Локальное тестирование LiteLLM Proxy (инструкция)

**Установка (без Docker):**
```bash
pip install 'litellm[proxy]'
```

**Запуск прокси:**
```bash
litellm --config docs/litellm/config.yaml --port 4000
```

**Тестовый запрос (primary провайдер):**
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

**Эмуляция отказа primary (неверный API-ключ):**
1. Временно испортить `OPENAI_API_KEY` в `.env`.
2. Перезапустить LiteLLM Proxy.
3. Отправить запрос — LiteLLM автоматически переключится на следующего провайдера в цепочке fallback.

**Статус:** LiteLLM Proxy **не поднят в проде**. Конфигурационный файл подготовлен, но интеграция с `ChatService` не выполнена. Сейчас `ChatService` обращается напрямую к OpenAI API.

---

## 8. Спецификация сетевых вызовов

| Категория | Операция | Ожидаемая задержка | Статус |
|-----------|----------|-------------------|--------|
| **Latency-Critical** | Чтение истории из JSONL (последние N строк) | < 5 мс | Реализовано |
| **Latency-Critical** | Поиск по `bugs_database.json` | < 10 мс | Реализовано |
| **Latency-Critical** | Проверка лимитов в Redis (Rate Limiter) | < 5 мс | **Не реализовано** |
| **Cost-Critical** | Вызов OpenAI API (`gpt-4o-mini`) | 200–400 мс TTFT | Реализовано (прямой вызов) |
| **Cost-Critical** | Вызов через LiteLLM Proxy | 200–400 мс TTFT + ~5 мс прокси | **Не реализовано** |
| **Cost-Critical** | Запись сообщения в JSONL (append-only) | < 5 мс | Реализовано |
| **Cost-Critical** | Фиксация транзакций в PostgreSQL | < 20 мс | **Не в проде** |

---

## 9. Матрица компонентов: статус реализации

| Компонент | Статус | Примечание |
|-----------|--------|------------|
| `JsonChatRepository` | Реализовано | Append-only JSONL, `messages.jsonl` |
| `PostgresChatRepository` | Готов, не подключен | SQLAlchemy 2.x async, ждет Docker/инфраструктуру |
| Redis Cache | Реализовано | Опционально, TTL при `temperature=0` |
| SSE Streaming | Реализовано | Guard-middleware обходит баг Starlette |
| Query Expansion | Реализовано | 3 синонима через `gpt-4o-mini` |
| Локальный поиск багов | Реализовано | `bugs_database.json` |
| Fake Tool-Calling | Реализовано | Assistant + tool сообщения |
| `fit_to_budget` | Реализовано | Токенизация `tiktoken` + обрезка |
| Nginx Reverse Proxy | **Не реализовано** | Нет reverse proxy в текущей сборке |
| Auth Middleware | **Не реализовано** | Нет аутентификации на уровне Gateway |
| Rate Limiter | **Не реализовано** | Нет ограничения RPM |
| LiteLLM Proxy | **Не реализовано** | Конфиг подготовлен, интеграция не выполнена |
| Circuit Breaker | **Не реализовано** | Будет через LiteLLM |
| OpenRouter Fallback | **Не реализовано** | Нет API-ключа, нет интеграции |
| Ollama Local Fallback | **Не реализовано** | Нет развернутого Ollama |
| Template-ответы при отказе LLM | **Не реализовано** | Нет локальной базы FAQ-заглушек |

---

## 10. Техдолг

| Проблема | Статус | Приоритет |
|----------|--------|-----------|
| Дублирующий роутер `/chat` | В процессе удаления | Средний |
| Захардкоженная дата в системном промпте (2026-06-02) | Нужно исправить | Средний |
| Postgres не в проде | Нет инфраструктуры/Docker | Низкий |
| Docker-сборка не работает | Проблемы с Docker Hub/правами | Низкий |
| LiteLLM Proxy не интегрирован | Требует поднятия сервиса + переписывания `ChatService` | Высокий |
| Nginx + Rate Limiter не подняты | Требует инфраструктуры | Средний |
