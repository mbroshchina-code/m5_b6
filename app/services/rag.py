"""RAG баг-ассистента на LlamaIndex."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from opentelemetry import trace
from app.observability.pii import redact_pii

import httpx
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.readers.file import FlatReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings
from app.core.rag_prompts import FALLBACK_ANSWER, build_system_prompt
from app.services.embeddings import EmbeddingService, EmbeddingSettings

import re
from threading import RLock

import structlog
from filelock import FileLock
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.core.rag_prompts import build_citation_prompt
from app.services.reranker import Reranker

PROJECT_ROOT = Path(__file__).resolve().parents[2]


QA_PROMPT = PromptTemplate(
    """
НАЙДЕННЫЕ ФРАГМЕНТЫ ЛОКАЛЬНОЙ БАЗЫ БАГОВ:
---------------------
{context_str}
---------------------

ЗАПРОС ОПЕРАТОРА:
{query_str}

Выполни классификацию багов по правилам системного промпта.
Верни только предусмотренный им Markdown:
разделы, номера, наименования, статусы, причины и временные решения.
Если подходящих багов нет, верни:
Подходящих багов не найдено
"""
)


REFINE_PROMPT = PromptTemplate(
    """
ЗАПРОС ОПЕРАТОРА:
{query_str}

ПРЕДВАРИТЕЛЬНЫЙ РЕЗУЛЬТАТ КЛАССИФИКАЦИИ:
{existing_answer}

ДОПОЛНИТЕЛЬНЫЕ ФРАГМЕНТЫ БАЗЫ:
---------------------
{context_msg}
---------------------

Уточни результат с учётом дополнительных фрагментов.
Соблюдай системные правила оценки, группировки и Markdown-формат.
Не дублируй номера багов и не придумывай сведения.
Если дополнительные фрагменты не подходят, сохрани прежний результат.
Верни только итоговый результат классификации.
"""
)


class CachedEmbedding(BaseEmbedding):
    """Адаптер сервиса Б5.1 к интерфейсу эмбеддингов LlamaIndex."""

    _service: EmbeddingService = PrivateAttr()

    def __init__(self, service: EmbeddingService) -> None:
        super().__init__(
            model_name=service.settings.model,
            embed_batch_size=service.settings.batch_size,
        )
        self._service = service

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._service.embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._service.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._service.embed_documents(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_settings = EmbeddingSettings()

        self._qdrant = None
        self._embeddings = None
        self._llm_http = None
        self._engine = None

    def _validate_settings(self) -> None:
        settings = self.settings
        embedding = self.embedding_settings

        collections = {
            settings.qdrant_collection,
            settings.rag_collection,
            settings.rag_baremetal_collection,
        }

        if len(collections) != 3:
            raise ValueError(
                "QDRANT_COLLECTION, RAG_COLLECTION и "
                "RAG_BAREMETAL_COLLECTION должны различаться"
            )

        if settings.rag_chunk_overlap >= settings.rag_chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP должен быть меньше RAG_CHUNK_SIZE"
            )

        if embedding.provider != "openai":
            raise ValueError(
                "Эта версия RAG рассчитана на EMBEDDING_PROVIDER=openai"
            )

        if embedding.dimensions != settings.embedding_dim:
            raise ValueError(
                "EMBEDDING_DIMENSIONS должен совпадать с EMBEDDING_DIM"
            )

        if not settings.llm.openai_api_key.get_secret_value():
            raise ValueError("Не указан ключ OpenAI для генерации ответа")

    def _load_documents(self):
        corpus_dir = self.settings.rag_corpus_dir

        if not corpus_dir.is_absolute():
            corpus_dir = PROJECT_ROOT / corpus_dir

        if not corpus_dir.is_dir():
            raise ValueError(f"Папка корпуса не найдена: {corpus_dir}")

        extensions = {".md", ".txt", ".pdf", ".docx"}
        files = sorted(
            path
            for path in corpus_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

        if len(files) != 10:
            raise ValueError(
                f"В корпусе должно быть 10 документов, найдено {len(files)}"
            )

        # Отпечаток защищает от подключения к индексу другого корпуса
        # или индексу, построенному с другими параметрами.
        signature_data = {
            "schema": 1,
            "model": self.embedding_settings.model,
            "provider": self.embedding_settings.provider,
            "embedding_base_url": self.embedding_settings.openai_base_url,
            "dimension": self.settings.embedding_dim,
            "chunk_size": self.settings.rag_chunk_size,
            "chunk_overlap": self.settings.rag_chunk_overlap,
            "files": [
                {
                    "path": path.relative_to(corpus_dir).as_posix(),
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

        documents = SimpleDirectoryReader(
            input_dir=str(corpus_dir),
            recursive=True,
            required_exts=sorted(extensions),
            # Сохраняем Markdown-файл целиком до SentenceSplitter:
            # заголовки не должны оторвать решение от описания бага.
            file_extractor={
                ".md": FlatReader(),
                ".txt": FlatReader(),
            },
        ).load_data()

        for number, document in enumerate(documents):
            filename = document.metadata.get("file_name")

            if not filename:
                raise ValueError("Ридер не сохранил file_name документа")

            document.id_ = str(
                uuid5(NAMESPACE_URL, f"{signature}/document/{number}")
            )
            document.metadata["rag_signature"] = signature

            # Технические метаданные не отправляем в эмбеддинг или LLM.
            excluded = [
                key for key in document.metadata if key != "file_name"
            ]
            document.excluded_embed_metadata_keys = excluded
            document.excluded_llm_metadata_keys = excluded

        return documents, signature

    def _check_collection(self, signature: str) -> int:
        name = self.settings.rag_collection

        if not self._qdrant.collection_exists(collection_name=name):
            return 0

        info = self._qdrant.get_collection(collection_name=name)
        vectors = info.config.params.vectors

        params = (
            vectors.get("text-dense")
            if isinstance(vectors, dict)
            else vectors
        )

        if not isinstance(params, VectorParams):
            raise ValueError("Неизвестный формат векторов RAG-коллекции")

        if (
            params.size != self.settings.embedding_dim
            or params.distance != Distance.COSINE
        ):
            raise ValueError(
                "Размерность или метрика существующей RAG-коллекции "
                "не совпадают с настройками"
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
                    or "_node_content" not in payload
                ):
                    raise ValueError(
                        "RAG-коллекция содержит другой корпус, другие "
                        "настройки или чужой формат данных. "
                        "Укажите новое имя RAG_COLLECTION. "
                        "Существующие данные не изменены."
                    )

            count += len(records)

            if offset is None:
                return count

    def build(self) -> None:
        """Построить индекс один раз либо подключиться к готовому."""

        if self._engine is not None:
            return

        self._validate_settings()
        documents, signature = self._load_documents()

        settings = self.settings

        try:
            key = (
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None
            )

            self._qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=key or None,
                timeout=60,
                trust_env=False,
            )

            existing_count = self._check_collection(signature)

            self._embeddings = EmbeddingService(self.embedding_settings)
            embed_model = CachedEmbedding(self._embeddings)

            llm_settings = settings.llm

            if llm_settings.use_litellm_proxy:
                base_url = llm_settings.litellm_proxy_url
                proxy_url = None
            else:
                base_url = llm_settings.base_url
                proxy_url = llm_settings.openai_proxy_url

            self._llm_http = httpx.Client(
                proxy=proxy_url or None,
                timeout=llm_settings.request_timeout,
            )

            llm = LlamaOpenAI(
                model=llm_settings.default_model,
                api_key=llm_settings.openai_api_key.get_secret_value(),
                api_base=base_url,
                http_client=self._llm_http,
                timeout=llm_settings.request_timeout,
                max_retries=llm_settings.max_retries,
                temperature=0,
                system_prompt=build_system_prompt(),
            )

            splitter = SentenceSplitter(
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
                id_func=lambda index, document: str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{signature}/{document.id_}/chunk/{index}",
                    )
                ),
            )

            expected_count = len(
                splitter.get_nodes_from_documents(documents)
            )

            if expected_count < settings.rag_similarity_top_k:
                raise ValueError(
                    "В корпусе недостаточно фрагментов для заданного top_k"
                )

            if existing_count > expected_count:
                raise ValueError(
                    "В коллекции больше точек, чем ожидается. "
                    "Проверьте её содержимое перед продолжением."
                )

            vector_store = QdrantVectorStore(
                client=self._qdrant,
                collection_name=settings.rag_collection,
                dense_vector_name="text-dense",
                dense_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
                batch_size=128,
                enable_hybrid=False,
            )

            if existing_count == expected_count:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model,
                )
                print(
                    f"RAG: подключён готовый индекс, "
                    f"фрагментов: {existing_count}"
                )
            else:
                storage = StorageContext.from_defaults(
                    vector_store=vector_store,
                )

                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage,
                    transformations=[splitter],
                    embed_model=embed_model,
                    show_progress=True,
                )

                # Проверяем завершённость индексации.
                actual_count = self._qdrant.count(
                    collection_name=settings.rag_collection,
                    exact=True,
                ).count

                if actual_count != expected_count:
                    raise RuntimeError(
                        f"Индексация неполная: {actual_count} фрагментов "
                        f"вместо {expected_count}. Повторите запуск."
                    )

                print(f"RAG: индекс построен, фрагментов: {actual_count}")

            self._engine = index.as_query_engine(
                llm=llm,
                similarity_top_k=settings.rag_similarity_top_k,
                response_mode="compact",
                text_qa_template=QA_PROMPT,
                refine_template=REFINE_PROMPT,
            )

        except Exception:
            self.close()
            raise

    def prepare_context(self, question: str) -> dict:
        """Поиск карточек без вызова модели генерации ответа.

        Может обращаться к embedding API.
        Возвращает контекст, источники и решение score-guard.
        """
        with self._lock:
            return self._prepare_context(question)

    def _prepare_context(self, question: str) -> dict:
        if self._index is None:
            raise RuntimeError("Сначала вызовите build()")

        question = question.strip()

        if not question:
            raise ValueError("Пустой запрос")

        # Пока ingestion обновляет индекс, не читаем смесь версий.
        with FileLock(
            str(self.state_dir / "ingest.lock"),
            timeout=5,
        ):
            candidates = self._retrieve(question)

            top_score = max(
                (item["score"] for item in candidates),
                default=0.0,
            )

            if (
                not candidates
                or top_score < self.settings.rag_score_threshold
            ):
                self._log.info(
                    "rag_score_guard",
                    top_score=top_score,
                    threshold=self.settings.rag_score_threshold,
                    llm_called=False,
                )

                return {
                    "allowed": False,
                    "top_score": top_score,
                    "context": [],
                    "sources": [],
                }

            documents = self._read_documents(candidates)

        if self.settings.rag_reranker_enabled:
            if self._reranker is None:
                self._reranker = Reranker()

            candidates = self._reranker.rerank(
                question,
                candidates,
                top_n=self.settings.rag_context_top_n,
            )
        else:
            candidates = candidates[
                :self.settings.rag_context_top_n
            ]

        context, sources = self._make_context(
            candidates,
            documents,
        )

        self._log.info(
            "rag_context_prepared",
            collection=self.collection,
            selected_documents=len(candidates),
            sources_count=len(sources),
            top_score=top_score,
            generation_called=False,
        )

        return {
            "allowed": bool(sources),
            "top_score": top_score,
            "context": context,
            "sources": sources,
        }

    def answer(self, question: str) -> dict:
        """Найти фрагменты и сформировать ответ с источниками."""

        if self._engine is None:
            raise RuntimeError("Сначала вызовите RAGService.build()")

        question = question.strip()

        if not question:
            raise ValueError("Вопрос не должен быть пустым")

        bundle = QueryBundle(query_str=question)
        nodes = self._engine.retrieve(bundle)

        top_score = max(
            (
                float(node.score)
                for node in nodes
                if node.score is not None
            ),
            default=0.0,
        )

        # Возвращаем найденные фрагменты даже при fallback:
        # они нужны для оценки retrieval, а не как подтверждение ответа.
        sources = [
            {
                "text": node.node.get_content()[:300],
                "source": node.node.metadata.get("file_name"),
                "score": (
                    round(float(node.score), 3)
                    if node.score is not None
                    else None
                ),
            }
            for node in nodes
        ]

        if not nodes or top_score < self.settings.rag_min_score:
            answer = FALLBACK_ANSWER
        else:
            response = self._engine.synthesize(
                query_bundle=bundle,
                nodes=nodes,
            )
            answer = str(response).strip() or FALLBACK_ANSWER

        return {
            "answer": answer,
            "top_score": round(top_score, 3),
            "sources": sources,
        }

    def close(self) -> None:
        """Закрыть собственные клиенты и дисковый кеш."""

        self._engine = None

        try:
            if self._llm_http is not None:
                self._llm_http.close()
        finally:
            self._llm_http = None
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

class CitationRAGService(RAGService):
    """Поиск по готовой ingestion-коллекции с цитатами."""

    def __init__(self) -> None:
        super().__init__()
        self._index = None
        self._llm = None
        self._reranker = None
        self._lock = RLock()
        self._log = structlog.get_logger(__name__)

        self.collection = self.settings.ingest_collection
        self.state_dir = (
            PROJECT_ROOT
            / self.settings.ingest_state_dir
            / self.collection
        )

    def build(self) -> None:
        """Подключиться к существующему индексу. Не переиндексировать."""
        with self._lock:
            if self._index is not None:
                return

            try:
                self._build()
            except Exception:
                self.close()
                raise

    def _build(self) -> None:
        settings = self.settings
        embedding = self.embedding_settings

        if embedding.provider != "openai":
            raise ValueError(
                "Текущая ingestion-коллекция рассчитана на OpenAI embeddings"
            )

        if embedding.dimensions != settings.embedding_dim:
            raise ValueError(
                "EMBEDDING_DIMENSIONS не совпадает с EMBEDDING_DIM"
            )

        if settings.rag_context_top_n > settings.retrieval_top_k:
            raise ValueError(
                "RAG_CONTEXT_TOP_N не должен превышать RETRIEVAL_TOP_K"
            )

        config_path = self.state_dir / "configuration.json"
        store_path = self.state_dir / "docstore.json"

        if not config_path.exists() or not store_path.exists():
            raise RuntimeError(
                "Нет сохранённого состояния ingestion. "
                "Сначала выполните scripts/ingest.py data"
            )

        saved = json.loads(config_path.read_text(encoding="utf-8"))

        expected = {
            "collection": self.collection,
            "model": embedding.model,
            "dimension": settings.embedding_dim,
            "embedding_endpoint": embedding.openai_base_url,
        }

        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(
                    f"Конфигурация индекса не совпадает: {key}. "
                    "Нельзя искать другой embedding-моделью "
                    "в ранее построенной коллекции."
                )

        api_key = settings.qdrant_api_key
        self._qdrant = QdrantClient(
            url=settings.qdrant_url,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=60,
            trust_env=False,
        )

        if not self._qdrant.collection_exists(self.collection):
            raise RuntimeError(
                f"Коллекция {self.collection!r} отсутствует"
            )

        info = self._qdrant.get_collection(self.collection)
        vectors = info.config.params.vectors

        if not isinstance(vectors, dict) or "text-dense" not in vectors:
            raise ValueError(
                "В ingestion-коллекции ожидается вектор text-dense"
            )

        vector_config = vectors["text-dense"]

        if (
            vector_config.size != settings.embedding_dim
            or vector_config.distance != Distance.COSINE
        ):
            raise ValueError(
                "У коллекции неправильная размерность или метрика"
            )

        count = self._qdrant.count(
            collection_name=self.collection,
            exact=True,
        ).count

        if count == 0:
            raise RuntimeError("Ingestion-коллекция пустая")

        self._embeddings = EmbeddingService()
        embed_model = CachedEmbedding(self._embeddings)

        vector_store = QdrantVectorStore(
            client=self._qdrant,
            collection_name=self.collection,
            dense_vector_name="text-dense",
            enable_hybrid=False,
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        llm_settings = settings.llm

        if llm_settings.use_litellm_proxy:
            base_url = llm_settings.litellm_proxy_url
            proxy_url = None
        else:
            base_url = llm_settings.base_url
            proxy_url = llm_settings.openai_proxy_url

        self._llm_http = httpx.Client(
            proxy=proxy_url or None,
            timeout=llm_settings.request_timeout,
        )

        self._llm = LlamaOpenAI(
            model=llm_settings.default_model,
            api_key=llm_settings.openai_api_key.get_secret_value(),
            api_base=base_url,
            http_client=self._llm_http,
            timeout=llm_settings.request_timeout,
            max_retries=llm_settings.max_retries,
            temperature=0,
        )

        self._index = index

        self._log.info(
            "rag_citation_index_connected",
            collection=self.collection,
            points_count=count,
            retrieval_top_k=settings.retrieval_top_k,
            context_top_n=settings.rag_context_top_n,
            reranker_enabled=settings.rag_reranker_enabled,
        )

    def _retrieve(self, question: str) -> list[dict]:
        tracer = trace.get_tracer("bag-assistant.rag")

        with tracer.start_as_current_span(
            "RAG retrieval",
            attributes={
                "openinference.span.kind": "RETRIEVER",
                "input.value": redact_pii(question),
                "rag.collection": self.collection,
                "rag.top_k": self.settings.retrieval_top_k,
            },
        ) as span:
            candidates = self._retrieve_impl(question)

            span.set_attribute(
                "rag.documents_count",
                len(candidates),
            )

            for index, item in enumerate(candidates):
                prefix = f"retrieval.documents.{index}.document"

                span.set_attribute(
                    f"{prefix}.id",
                    str(item["document_id"]),
                )
                span.set_attribute(
                    f"{prefix}.content",
                    redact_pii(item["text"]),
                )
                span.set_attribute(
                    f"{prefix}.score",
                    float(item["score"]),
                )

                metadata = item.get("metadata") or {}
                safe_metadata = {
                    key: metadata[key]
                    for key in (
                        "file_name",
                        "bug_id",
                        "page",
                        "category",
                    )
                    if metadata.get(key) is not None
                }

                span.set_attribute(
                    f"{prefix}.metadata",
                    redact_pii(
                        json.dumps(
                            safe_metadata,
                            ensure_ascii=False,
                            default=str,
                        )
                    ),
                )

            return candidates

    def _retrieve_impl(self, question: str) -> list[dict]:
        """Получить top-K уникальных документов по лучшему чанку."""
        count = self._qdrant.count(
            collection_name=self.collection,
            exact=True,
        ).count

        if count == 0:
            return []

        top_k = self.settings.retrieval_top_k

        # Эмбеддинг вопроса рассчитываем один раз,
        # даже если потребуется расширить выборку чанков.
        bundle = QueryBundle(
            query_str=question,
            embedding=self._embeddings.embed_query(question),
        )

        limit = min(top_k * 2, count)

        while True:
            retriever = self._index.as_retriever(
                similarity_top_k=limit,
            )
            nodes = retriever.retrieve(bundle)

            unique = {}

            for item in sorted(
                nodes,
                key=lambda item: (
                    float(item.score)
                    if item.score is not None
                    else -1.0
                ),
                reverse=True,
            ):
                if item.score is None:
                    continue

                metadata = item.node.metadata
                document_id = (
                    item.node.ref_doc_id or metadata.get("doc_id")
                )

                if not document_id:
                    raise RuntimeError(
                        "У найденного чанка отсутствует ID документа"
                    )

                if document_id not in unique:
                    unique[document_id] = {
                        "document_id": document_id,
                        "text": item.node.get_content(),
                        "score": float(item.score),
                        "metadata": dict(metadata),
                    }

            if len(unique) >= top_k or limit >= count:
                return list(unique.values())[:top_k]

            limit = min(limit * 2, count)

    def _read_documents(self, candidates: list[dict]) -> dict:
        """Загрузить полные карточки из сохранённого docstore."""
        store = SimpleDocumentStore.from_persist_path(
            str(self.state_dir / "docstore.json")
        )
        documents = {}

        for candidate in candidates:
            document_id = candidate["document_id"]
            document = store.get_document(document_id)

            if document is None:
                raise RuntimeError(
                    "Документ найден в Qdrant, но отсутствует в docstore"
                )

            indexed_hash = candidate["metadata"].get("file_sha256")
            stored_hash = document.metadata.get("file_sha256")

            if not indexed_hash or indexed_hash != stored_hash:
                raise RuntimeError(
                    "Версии документа в Qdrant и docstore различаются. "
                    "Повторите ingestion."
                )

            documents[document_id] = document

        return documents

    @staticmethod
    def _make_context(candidates: list[dict], documents: dict):
        context = []
        sources = []

        for candidate in candidates:
            document = documents[candidate["document_id"]]
            metadata = document.metadata
            units = json.loads(document.text)

            for unit in units:
                text = unit["text"].strip()

                if not text:
                    continue

                citation_id = len(sources) + 1

                sources.append(
                    {
                        "id": citation_id,
                        "file_name": metadata["file_name"],
                        "page": unit.get("page"),
                        "score": round(candidate["score"], 3),
                        "snippet": text[:300],
                    }
                )

                context.append(
                    {
                        "citation_id": citation_id,
                        "file_name": metadata["file_name"],
                        "page": unit.get("page"),
                        "bug_id": metadata.get("bug_id"),
                        "bug_created_at": metadata.get("bug_created_at"),
                        "text": text,
                    }
                )

        return context, sources

    @staticmethod
    def _fallback(top_score: float) -> dict:
        return {
            "answer": FALLBACK_ANSWER,
            "top_score": round(top_score, 3),
            "confident": False,
            "sources": [],
        }

    def answer(self, question: str) -> dict:
        with self._lock:
            return self._answer(question)

    def _answer(self, question: str) -> dict:
        question = question.strip()
        prepared = self.prepare_context(question)

        top_score = prepared["top_score"]

        if not prepared["allowed"]:
            return self._fallback(top_score)

        context = prepared["context"]
        sources = prepared["sources"]

        response = self._llm.chat(
            [
                ChatMessage(
                    role="system",
                    content=build_citation_prompt(),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "КОНТЕКСТ БАЗЫ БАГОВ:\n"
                        + json.dumps(context, ensure_ascii=False)
                        + "\n\nЗАПРОС ОПЕРАТОРА:\n"
                        + question
                    ),
                ),
            ]
        )

        answer = (response.message.content or "").strip()

        if not answer:
            raise RuntimeError("Модель вернула пустой ответ")

        if answer.rstrip(".") == FALLBACK_ANSWER:
            return self._fallback(top_score)

        cited_ids = {
            int(value)
            for value in re.findall(r"\[(\d+)\]", answer)
        }
        allowed_ids = {source["id"] for source in sources}

        if not cited_ids or not cited_ids.issubset(allowed_ids):
            self._log.warning("rag_invalid_citations")
            raise RuntimeError(
                "Модель не указала корректные номера источников"
            )

        return {
            "answer": answer,
            "top_score": round(top_score, 3),
            "confident": True,
            "sources": sources,
        }

    def close(self) -> None:
        with self._lock:
            self._index = None
            self._llm = None
            self._reranker = None
            super().close()
                    
def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Проверка RAG баг-ассистента через LlamaIndex",
    )
    parser.add_argument(
        "--question",
        default=(
            "После выбора типа оплаты перед печатью чека "
            "долго крутится анимация загрузки."
        ),
        help="Вопрос оператора",
    )
    args = parser.parse_args()

    service = RAGService()

    try:
        service.build()
        result = service.answer(args.question)

        if not isinstance(result, dict):
            raise RuntimeError("RAG должен возвращать словарь")

        if not isinstance(result.get("answer"), str):
            raise RuntimeError("В результате отсутствует текст answer")

        sources = result.get("sources")
        if not isinstance(sources, list) or len(sources) < 3:
            raise RuntimeError("Ожидалось минимум три источника")

        for source in sources:
            if not isinstance(source, dict) or not {
                "text", "source", "score"
            }.issubset(source):
                raise RuntimeError(
                    "Каждый источник должен содержать text, source, score"
                )

        print(json.dumps(result, ensure_ascii=False, indent=2))

    finally:
        service.close()


if __name__ == "__main__":
    main()