"""Pydantic-схемы для чат-API."""

from pydantic import BaseModel


class Message(BaseModel):
    """Сообщение для LLM. content может быть строкой или списком content-parts."""
    role: str
    content: str | list[dict]


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, usage) -> "Usage":
        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )


class ChatDelta(BaseModel):
    content: str | None = None
    usage: Usage | None = None


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    finish_reason: str | None = None
    cached: bool = False

    @classmethod
    def from_openai(cls, raw) -> "ChatResponse":
        msg = raw.choices[0].message
        return cls(
            content=msg.content or "",
            model=raw.model,
            usage=Usage.from_openai(raw.usage),
            finish_reason=raw.choices[0].finish_reason,
        )
