"""Многоформатная индексация учебного корпуса Б5.5.

Поддерживаемые форматы этого этапа: Markdown и PDF.
Исходный JSON и коллекции предыдущих блоков не изменяются.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import pymupdf as fitz
from filelock import FileLock
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import MarkdownReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from app.core.config import get_settings
from app.services import chunking
from app.services.embeddings import EmbeddingService, EmbeddingSettings
from app.services.rag import CachedEmbedding


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = {".md", ".pdf"}
log = logging.getLogger(__name__)


def absolute_path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def save_json(path: Path, value: dict) -> None:
    """Сначала записать временный файл, затем заменить состояние."""
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")

    with temporary.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)

    temporary.replace(path)


class FileChunker(TransformComponent):
    """Semantic + ограничение длинных фрагментов через SentenceSplitter.

    PDF разбивается отдельно по страницам, чтобы сохранить точные
    номера страниц для будущих цитат.
    """

    chunk_size: int = 512
    chunk_overlap: int = 32
    _embedding: BaseEmbedding = PrivateAttr()

    def __init__(
        self,
        embedding: BaseEmbedding,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embedding = embedding

    def __call__(self, nodes, **kwargs):
        result = []

        for document in nodes:
            # Техническое представление страниц в docstore.
            # В embedding отправляется текст страницы, а не этот JSON.
            units = json.loads(document.text)

            for unit_number, unit in enumerate(units):
                metadata = dict(document.metadata)
                metadata["page"] = unit["page"]

                # Уникальность чанков, включая одинаковый текст
                # на разных страницах одного PDF.
                metadata["doc_id"] = (
                    f"{document.metadata['relative_path']}#part={unit_number}"
                )

                page_document = Document(
                    id_=document.id_,
                    text=unit["text"],
                    metadata=metadata,
                )

                chunks = chunking.semantic(
                    [page_document],
                    embed_model=self._embedding,
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )

                for node in chunks:
                    if chunking.count_tokens(node.text) > 8000:
                        raise ValueError("Получен слишком длинный чанк")

                    # Все страницы принадлежат одному исходному файлу.
                    node.metadata["doc_id"] = document.id_
                    node.excluded_embed_metadata_keys = list(node.metadata)
                    node.excluded_llm_metadata_keys = list(node.metadata)

                result.extend(chunks)

        return result


class IngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_settings = EmbeddingSettings()
        self.data_root = absolute_path(self.settings.ingest_data_root)

        self.collection = self.settings.ingest_collection
        if not re.fullmatch(r"ingest_[a-z0-9_]+", self.collection):
            raise ValueError("Имя новой коллекции должно начинаться с ingest_")

        protected = {
            self.settings.qdrant_collection,
            self.settings.rag_collection,
            self.settings.rag_baremetal_collection,
            self.settings.retrieval_collection,
        }
        if self.collection in protected:
            raise ValueError("Нельзя использовать существующую рабочую коллекцию")

        if self.settings.retrieval_strategy != "semantic_recursive":
            raise ValueError("Этот pipeline использует semantic_recursive")

        size = self.settings.retrieval_chunk_size
        overlap = self.settings.retrieval_chunk_overlap
        if not 0 <= overlap < size:
            raise ValueError("Нужно: chunk_size > chunk_overlap >= 0")

        if (
            self.embedding_settings.provider != "openai"
            or self.embedding_settings.model != "text-embedding-3-small"
            or self.embedding_settings.dimensions != self.settings.embedding_dim
        ):
            raise ValueError("Настройки эмбеддингов не соответствуют эксперименту")

        self.state_dir = (
            absolute_path(self.settings.ingest_state_dir) / self.collection
        )

    def _find_files(self, input_path: Path) -> list[Path]:
        input_path = absolute_path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Не найден путь: {input_path}")

        if not input_path.is_relative_to(self.data_root):
            raise ValueError("Индексировать можно только файлы внутри data")

        candidates = (
            [input_path]
            if input_path.is_file()
            else sorted(input_path.rglob("*"))
        )

        files = []

        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue

            resolved = path.resolve()
            if not resolved.is_relative_to(self.data_root):
                continue

            relative = resolved.relative_to(self.data_root)
            category = relative.parts[0] if len(relative.parts) > 1 else ""

            if category in self.settings.ingest_categories:
                files.append(resolved)

        if not files:
            raise ValueError(
                "Не найдено MD/PDF в разрешённых категориях "
                f"{self.settings.ingest_categories}"
            )

        return files

    def _read_file(self, path: Path) -> Document:
        relative = path.relative_to(self.data_root)
        source_bytes = path.read_bytes()
        stat = path.stat()
        author = None

        if path.suffix.lower() == ".pdf":
            # source_bytes уже прочитаны через path.read_bytes().
            # Парсер работает с памятью и не открывает исходный файл.
            with fitz.open(
                stream=source_bytes,
                filetype="pdf",
            ) as pdf:
                if not pdf.is_pdf:
                    raise ValueError("Файл не является PDF")

                if pdf.needs_pass:
                    raise ValueError("PDF защищён паролем")

                raw_author = (
                    (pdf.metadata or {}).get("author") or ""
                ).strip()

                if raw_author.lower() not in {
                    "",
                    "anonymous",
                    "(anonymous)",
                    "unspecified",
                }:
                    author = raw_author

                units = []

                for page in pdf:
                    text = page.get_text().strip()

                    if text:
                        units.append(
                            {
                                "page": page.number + 1,
                                "text": text,
                            }
                        )

        else:
            loaded = MarkdownReader().load_data(file=str(path))
            text = "\n\n".join(
                item.text for item in loaded if item.text.strip()
            ).strip()
            units = [{"page": None, "text": text}] if text else []

        if not units:
            raise ValueError(
                "Не извлечён текст. Для сканированного PDF потребуется OCR."
            )

        # Если файл редактировали во время чтения, не индексируем смесь версий.
        if path.read_bytes() != source_bytes:
            raise RuntimeError(
                f"Файл изменился во время чтения: {relative}. Повторите запуск."
            )

        metadata = {
            "source": path.name,
            "file_name": path.name,
            "relative_path": relative.as_posix(),
            "category": relative.parts[0] if len(relative.parts) > 1 else "",
            "last_modified": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }

        if author:
            metadata["author"] = author

        version = re.search(
            r"(?:^|[_-])v(\d+(?:\.\d+)*)",
            path.stem,
            flags=re.IGNORECASE,
        )
        if version:
            metadata["version"] = version.group(1)

        bug_id = re.fullmatch(r"bug_(\d+)", path.stem)
        if bug_id:
            metadata["bug_id"] = int(bug_id.group(1))

        full_text = "\n".join(unit["text"] for unit in units)
        bug_date = re.search(
            r"Дата создания(?: бага)?[^\d]{0,100}(\d{4}-\d{2}-\d{2})",
            full_text,
        )
        if bug_date:
            metadata["bug_created_at"] = bug_date.group(1)

        return Document(
            id_=str(uuid5(NAMESPACE_URL, f"bag-files/{relative.as_posix()}")),
            text=json.dumps(units, ensure_ascii=False, sort_keys=True),
            metadata=metadata,
            excluded_embed_metadata_keys=list(metadata),
            excluded_llm_metadata_keys=list(metadata),
        )

    def _read_files(self, files, rename_failed: bool):
        documents = []
        failed = 0

        for path in files:
            try:
                documents.append(self._read_file(path))
            except (
                ValueError,
                UnicodeError,
                OSError,
                fitz.FileDataError,
                fitz.EmptyFileError,
            ) as exc:
                failed += 1

                if rename_failed:
                    target = path.with_name(path.name + ".failed")
                    if target.exists():
                        target = path.with_name(
                            f"{path.name}.{uuid4().hex}.failed"
                        )

                    path.rename(target)
                    log.error(
                        "file_parse_failed file=%s error_type=%s renamed_to=%s",
                        path.name,
                        type(exc).__name__,
                        target.name,
                    )
                else:
                    log.error(
                        "file_parse_failed file=%s error_type=%s",
                        path.name,
                        type(exc).__name__,
                    )

        return documents, failed

    def _configuration(self) -> dict:
        return {
            "schema_version": 1,
            "collection": self.collection,
            "model": self.embedding_settings.model,
            "dimension": self.settings.embedding_dim,
            "embedding_endpoint": self.embedding_settings.openai_base_url,
            "strategy": "semantic_recursive",
            "chunk_size": self.settings.retrieval_chunk_size,
            "chunk_overlap": self.settings.retrieval_chunk_overlap,
            "buffer_size": 1,
            "breakpoint_percentile_threshold": 95,
            "chunking_code_sha256": hashlib.sha256(
                Path(chunking.__file__).read_bytes()
            ).hexdigest(),
        }

    def _persist_docstore(self, store) -> None:
        destination = self.state_dir / "docstore.json"
        temporary = self.state_dir / f"docstore.{uuid4().hex}.tmp"
        store.persist(persist_path=str(temporary))

        for attempt in range(10):
            try:
                temporary.replace(destination)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.5)

    def _prepare_storage(self, client):
        config_path = self.state_dir / "configuration.json"
        docstore_path = self.state_dir / "docstore.json"
        configuration = self._configuration()

        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if saved != configuration:
                raise ValueError(
                    "Изменилась конфигурация индексации. "
                    "Для нового эксперимента выберите новое INGEST_COLLECTION. "
                    "Существующие данные не удалены."
                )

        store = (
            SimpleDocumentStore.from_persist_path(str(docstore_path))
            if docstore_path.exists()
            else SimpleDocumentStore()
        )

        if client.collection_exists(self.collection):
            info = client.get_collection(self.collection)
            vectors = info.config.params.vectors
            dense = vectors.get("text-dense") if isinstance(vectors, dict) else None

            if (
                dense is None
                or dense.size != self.settings.embedding_dim
                or dense.distance != Distance.COSINE
            ):
                raise ValueError("Коллекция имеет другую схему векторов")

            count = client.count(
                collection_name=self.collection,
                exact=True,
            ).count

            if count and (not config_path.exists() or not docstore_path.exists()):
                raise ValueError(
                    "Коллекция заполнена, но её локальное состояние отсутствует. "
                    "Не продолжаем, чтобы не смешать индексы."
                )
        else:
            if store.get_all_document_hashes():
                raise ValueError(
                    "Docstore заполнен, но коллекция Qdrant отсутствует. "
                    "Нужно восстановить согласованное состояние."
                )

            client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "text-dense": VectorParams(
                        size=self.settings.embedding_dim,
                        distance=Distance.COSINE,
                    )
                },
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )

        for field in ("doc_id", "relative_path", "category"):
            client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )

        client.create_payload_index(
            collection_name=self.collection,
            field_name="last_modified",
            field_schema=PayloadSchemaType.DATETIME,
            wait=True,
        )

        if not config_path.exists():
            save_json(config_path, configuration)

        if not docstore_path.exists():
            self._persist_docstore(store)

        return store

    def _document_count(self, client, document_id: str) -> int:
        return client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        ).count

    def _index(self, documents) -> dict:
        api_key = self.settings.qdrant_api_key
        client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=120,
            trust_env=False,
        )
        embeddings = None

        try:
            store = self._prepare_storage(client)
            pipeline = None
            changed = 0
            unchanged = 0

            for document in documents:
                if store.get_document_hash(document.id_) == document.hash:
                    if self._document_count(client, document.id_) == 0:
                        raise RuntimeError(
                            "Документ есть в docstore, но его чанки пропали "
                            f"из Qdrant: {document.metadata['relative_path']}"
                        )

                    unchanged += 1
                    continue

                if pipeline is None:
                    embeddings = EmbeddingService(self.embedding_settings)
                    adapter = CachedEmbedding(embeddings)

                    vector_store = QdrantVectorStore(
                        client=client,
                        collection_name=self.collection,
                        dense_vector_name="text-dense",
                        enable_hybrid=False,
                        batch_size=128,
                    )

                    pipeline = IngestionPipeline(
                        transformations=[
                            FileChunker(
                                embedding=adapter,
                                chunk_size=self.settings.retrieval_chunk_size,
                                chunk_overlap=self.settings.retrieval_chunk_overlap,
                            ),
                            adapter,
                        ],
                        docstore=store,
                        docstore_strategy=DocstoreStrategy.UPSERTS,
                        vector_store=vector_store,
                        # Используем кеш EmbeddingService.
                        # Второй кеш трансформаций здесь не нужен.
                        disable_cache=True,
                    )

                original_hash = document.hash

                # При сбое API или Qdrant выполнение остановится.
                # Такой сбой не переименовывает исходный файл в .failed.
                nodes = pipeline.run(
                    documents=[document],
                    show_progress=False,
                )

                if not nodes:
                    raise RuntimeError("Изменённый документ не дал чанков")

                count = self._document_count(client, document.id_)
                if count != len(nodes):
                    raise RuntimeError(
                        f"Неполная запись {document.metadata['relative_path']}: "
                        f"{count} вместо {len(nodes)} чанков"
                    )

                # Фиксируем состояние только после успешной записи в Qdrant.
                store.set_document_hash(document.id_, original_hash)
                self._persist_docstore(store)
                changed += 1

                print(
                    f"Indexed: {document.metadata['relative_path']} "
                    f"({len(nodes)} chunks)",
                    flush=True,
                )

            count = client.count(
                collection_name=self.collection,
                exact=True,
            ).count

            return {
                "changed": changed,
                "unchanged": unchanged,
                "points_count": count,
                "collection": self.collection,
            }

        finally:
            try:
                if embeddings is not None:
                    embeddings.close()
            finally:
                client.close()

    def run(self, input_path: Path, check_only: bool = False) -> dict:
        files = self._find_files(input_path)

        if check_only:
            documents, failed = self._read_files(
                files,
                rename_failed=False,
            )
            return {
                "mode": "check",
                "files": len(files),
                "readable": len(documents),
                "failed": failed,
                "collection": self.collection,
            }

        self.state_dir.mkdir(parents=True, exist_ok=True)

        with FileLock(str(self.state_dir / "ingest.lock"), timeout=5):
            documents, failed = self._read_files(
                files,
                rename_failed=True,
            )

            if not documents:
                raise RuntimeError("Нет документов, пригодных для индексации")

            result = self._index(documents)
            result["failed"] = failed
            result["files"] = len(files)
            return result