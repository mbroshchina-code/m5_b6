"""RAG без LlamaIndex: файлы, чанкинг, Qdrant и прямой вызов LLM."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings
from app.core.rag_prompts import FALLBACK_ANSWER, build_system_prompt
from app.services.embeddings import EmbeddingService, EmbeddingSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Наивное разбиение по границам слов с ограничением в токенах."""

    encoding = tiktoken.get_encoding("cl100k_base")
    words = re.findall(r"\S+\s*", text)
    chunks = []
    start = 0

    def token_count(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    while start < len(words):
        end = start
        chunk = ""

        while end < len(words):
            candidate = chunk + words[end]

            if token_count(candidate) > chunk_size:
                break

            chunk = candidate
            end += 1

        if end == start:
            raise ValueError(
                "В документе есть слово или непрерывный фрагмент "
                "длиннее RAG_CHUNK_SIZE"
            )

        chunks.append(chunk.strip())

        if end == len(words):
            break

        # Последние слова предыдущего чанка попадут в следующий.
        next_start = end

        while next_start > start:
            candidate = "".join(words[next_start - 1 : end])

            if token_count(candidate) > overlap:
                break

            next_start -= 1

        start = max(start + 1, next_start)

    return chunks


class BareMetalRAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_settings = EmbeddingSettings()

        self._qdrant = None
        self._embeddings = None
        self._llm = None
        self._http = None
        self._ready = False

    def _prepare_chunks(self) -> tuple[list[dict], str]:
        settings = self.settings
        embedding = self.embedding_settings

        if len({
            settings.qdrant_collection,
            settings.rag_collection,
            settings.rag_baremetal_collection,
        }) != 3:
            raise ValueError("Имена трёх коллекций должны различаться")

        if settings.rag_chunk_overlap >= settings.rag_chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP должен быть меньше RAG_CHUNK_SIZE"
            )

        if embedding.provider != "openai":
            raise ValueError("Для сравнения используйте провайдера openai")

        if embedding.dimensions != settings.embedding_dim:
            raise ValueError(
                "EMBEDDING_DIMENSIONS должен совпадать с EMBEDDING_DIM"
            )

        if not settings.llm.openai_api_key.get_secret_value():
            raise ValueError("Не указан ключ OpenAI для генерации")

        corpus = settings.rag_corpus_dir

        if not corpus.is_absolute():
            corpus = PROJECT_ROOT / corpus

        supported = {".md", ".txt", ".pdf", ".docx"}
        files = sorted(
            path
            for path in corpus.rglob("*")
            if path.is_file() and path.suffix.lower() in supported
        )

        if len(files) != 10:
            raise ValueError(
                f"Ожидалось 10 документов, найдено {len(files)}"
            )

        # Для текущего эксперимента корпус полностью состоит из Markdown.
        if any(path.suffix.lower() not in {".md", ".txt"} for path in files):
            raise ValueError(
                "Bare-metal ридер сейчас поддерживает MD/TXT. "
                "Для PDF/DOCX потребуется добавить извлечение текста."
            )

        signature_data = {
            "schema": "baremetal-1",
            "model": embedding.model,
            "provider": embedding.provider,
            "embedding_base_url": embedding.openai_base_url,
            "dimension": settings.embedding_dim,
            "chunk_size": settings.rag_chunk_size,
            "overlap": settings.rag_chunk_overlap,
            "files": [
                {
                    "path": path.relative_to(corpus).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in files
            ],
        }

        signature = hashlib.sha256(
            json.dumps(
                signature_data,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        chunks = []

        for path in files:
            text = path.read_text(encoding="utf-8-sig")

            parts = split_text(
                text,
                chunk_size=settings.rag_chunk_size,
                overlap=settings.rag_chunk_overlap,
            )

            if not parts:
                raise ValueError(f"Документ пуст: {path.name}")

            relative_path = path.relative_to(corpus).as_posix()

            for number, part in enumerate(parts):
                chunks.append(
                    {
                        "id": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"{signature}/{relative_path}/{number}",
                            )
                        ),
                        "payload": {
                            "text": part,
                            "file_name": path.name,
                            "source": relative_path,
                            "chunk_index": number,
                            "rag_signature": signature,
                            "embedding_model": embedding.model,
                        },
                    }
                )

        if len(chunks) < settings.rag_similarity_top_k:
            raise ValueError("Недостаточно фрагментов для заданного top_k")

        return chunks, signature

    def _ensure_collection(self, signature: str) -> int:
        name = self.settings.rag_baremetal_collection

        if not self._qdrant.collection_exists(collection_name=name):
            self._qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )
            return 0

        info = self._qdrant.get_collection(collection_name=name)
        params = info.config.params.vectors

        if (
            not isinstance(params, VectorParams)
            or params.size != self.settings.embedding_dim
            or params.distance != Distance.COSINE
        ):
            raise ValueError(
                "Размерность или метрика bare-metal коллекции "
                "не соответствует настройкам"
            )

        count = 0
        offset = None

        while True:
            records, offset = self._qdrant.scroll(
                collection_name=name,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = record.payload or {}

                if (
                    payload.get("rag_signature") != signature
                    or not isinstance(payload.get("text"), str)
                ):
                    raise ValueError(
                        "В коллекции другой корпус или настройки. "
                        "Укажите новое имя RAG_BAREMETAL_COLLECTION. "
                        "Существующие точки не изменены."
                    )

            count += len(records)

            if offset is None:
                return count

    def build(self) -> None:
        """Создать индекс или подключиться к уже готовому."""

        if self._ready:
            return

        chunks, signature = self._prepare_chunks()
        settings = self.settings

        try:
            api_key = (
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None
            )

            self._qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=api_key or None,
                timeout=60,
                trust_env=False,
            )

            count = self._ensure_collection(signature)

            if count > len(chunks):
                raise ValueError(
                    "В bare-metal коллекции больше точек, чем ожидается"
                )

            self._embeddings = EmbeddingService(self.embedding_settings)

            llm = settings.llm
            base_url = (
                llm.litellm_proxy_url
                if llm.use_litellm_proxy
                else llm.base_url
            )
            proxy_url = (
                None if llm.use_litellm_proxy else llm.openai_proxy_url
            )

            self._http = httpx.Client(
                proxy=proxy_url or None,
                timeout=llm.request_timeout,
            )

            self._llm = OpenAI(
                api_key=llm.openai_api_key.get_secret_value(),
                base_url=base_url,
                http_client=self._http,
                max_retries=llm.max_retries,
            )

            if count == len(chunks):
                print(
                    f"Bare-metal: подключён готовый индекс, "
                    f"фрагментов: {count}"
                )
            else:
                for start in range(0, len(chunks), 128):
                    batch = chunks[start : start + 128]
                    texts = [item["payload"]["text"] for item in batch]
                    vectors = self._embeddings.embed_documents(texts)

                    points = []

                    for item, vector in zip(batch, vectors, strict=True):
                        if len(vector) != settings.embedding_dim:
                            raise ValueError(
                                f"Получена размерность {len(vector)}, "
                                f"ожидалась {settings.embedding_dim}"
                            )

                        points.append(
                            PointStruct(
                                id=item["id"],
                                vector=vector,
                                payload=item["payload"],
                            )
                        )

                    self._qdrant.upsert(
                        collection_name=settings.rag_baremetal_collection,
                        points=points,
                        wait=True,
                    )

                actual = self._qdrant.count(
                    collection_name=settings.rag_baremetal_collection,
                    exact=True,
                ).count

                if actual != len(chunks):
                    raise RuntimeError(
                        f"Неполная индексация: {actual} вместо {len(chunks)}"
                    )

                print(f"Bare-metal: индекс построен, фрагментов: {actual}")

            self._ready = True

        except Exception:
            self.close()
            raise

    def answer(self, question: str) -> dict:
        if not self._ready:
            raise RuntimeError("Сначала вызовите build()")

        question = question.strip()

        if not question:
            raise ValueError("Вопрос не должен быть пустым")

        query_vector = self._embeddings.embed_query(question)

        if len(query_vector) != self.settings.embedding_dim:
            raise ValueError("Неверная размерность вектора запроса")

        result = self._qdrant.query_points(
            collection_name=self.settings.rag_baremetal_collection,
            query=query_vector,
            limit=self.settings.rag_similarity_top_k,
            with_payload=True,
            with_vectors=False,
        )

        points = result.points
        top_score = max((float(p.score) for p in points), default=0.0)

        sources = [
            {
                "text": (point.payload or {}).get("text", "")[:300],
                "source": (point.payload or {}).get("file_name"),
                "score": round(float(point.score), 3),
            }
            for point in points
        ]

        if not points or top_score < self.settings.rag_min_score:
            answer = FALLBACK_ANSWER
        else:
            context = "\n\n---\n\n".join(
                f"Файл: {(point.payload or {}).get('file_name')}\n"
                f"{(point.payload or {}).get('text', '')}"
                for point in points
            )

            user_prompt = (
                "НАЙДЕННЫЕ ФРАГМЕНТЫ ЛОКАЛЬНОЙ БАЗЫ БАГОВ:\n"
                f"{context}\n\n"
                "ЗАПРОС ОПЕРАТОРА:\n"
                f"{question}\n\n"
                "Выполни классификацию по системным правилам. "
                "Верни только предусмотренный Markdown-формат. "
                f"Если подходящих багов нет, верни: {FALLBACK_ANSWER}"
            )

            response = self._llm.chat.completions.create(
                model=self.settings.llm.default_model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            answer = (
                response.choices[0].message.content or ""
            ).strip() or FALLBACK_ANSWER

        return {
            "answer": answer,
            "top_score": round(top_score, 3),
            "sources": sources,
        }

    def close(self) -> None:
        self._ready = False

        try:
            if self._llm is not None:
                self._llm.close()
        finally:
            self._llm = None
            try:
                if self._http is not None:
                    self._http.close()
            finally:
                self._http = None
                try:
                    if self._embeddings is not None:
                        self._embeddings.close()
                finally:
                    self._embeddings = None
                    try:
                        if self._qdrant is not None:
                            self._qdrant.close()
                    finally:
                        self._qdrant = None