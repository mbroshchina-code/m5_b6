"""Стратегии разбиения карточек багов для эксперимента Б5.4.

Модуль:
- не читает и не изменяет bugs_database.json;
- не создаёт коллекции Qdrant;
- не вызывает модель, формирующую ответы оператору.

Только semantic использует переданную embedding-модель
для определения смысловых границ.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

import tiktoken
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import (
    Document,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from razdel import sentenize


# При изменении алгоритма увеличиваем версию,
# чтобы различать результаты разных реализаций.
CHUNKING_VERSION = "1"


@lru_cache(maxsize=1)
def _encoding():
    """Токенизатор для выбранной text-embedding-3-small."""
    return tiktoken.get_encoding("cl100k_base")


def tokenize(text: str) -> list[int]:
    """Одинаковый подсчёт токенов во всём эксперименте."""
    return _encoding().encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def russian_sentences(text: str) -> list[str]:
    """Разделить русский текст, сохранив пробелы между предложениями.

    SemanticSplitter соединяет предложения без собственного разделителя,
    поэтому пробелы и переводы строк сохраняем в исходных фрагментах.
    """
    if not text.strip():
        return []

    sentences = list(sentenize(text))

    if not sentences:
        return [text]

    parts: list[str] = []

    for index, sentence in enumerate(sentences):
        start = 0 if index == 0 else sentence.start
        end = (
            sentences[index + 1].start
            if index + 1 < len(sentences)
            else len(text)
        )
        parts.append(text[start:end])

    return parts


def _validate_size(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше нуля")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap должен быть не меньше нуля "
            "и строго меньше chunk_size"
        )


def _make_nodes(
    documents: list[Document],
    *,
    strategy: str,
    parameters: dict,
    split: Callable[[str], list[str]],
) -> list[TextNode]:
    """Обрабатывать каждый баг отдельно и сохранять его идентификатор.

    Загрузчик должен передать metadata["doc_id"], например bug_174.md.
    Это идентификатор для golden dataset, а не обязательный файл на диске.
    """
    prepared: list[tuple[Document, str, str]] = []
    seen_ids: set[str] = set()

    # Сначала проверяем весь вход, до возможных платных вызовов semantic.
    for document in documents:
        doc_id = document.metadata.get("doc_id")

        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(
                "У каждого документа должен быть непустой "
                'metadata["doc_id"], например "bug_174.md"'
            )

        if doc_id in seen_ids:
            raise ValueError(f"Повторяющийся doc_id: {doc_id}")

        seen_ids.add(doc_id)

        text = document.get_content(metadata_mode=MetadataMode.NONE)

        if not text.strip():
            raise ValueError(f"Пустой текст документа: {doc_id}")

        prepared.append((document, doc_id, text))

    nodes: list[TextNode] = []

    for document, doc_id, text in prepared:
        pieces = split(text)

        if not pieces or any(not piece.strip() for piece in pieces):
            raise ValueError(
                f"Стратегия {strategy} вернула пустой чанк для {doc_id}"
            )

        for chunk_index, piece in enumerate(pieces):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunking_strategy": strategy,
                    "chunking_version": CHUNKING_VERSION,
                    "chunk_tokens": count_tokens(piece),
                }
            )

            identity = json.dumps(
                {
                    "version": CHUNKING_VERSION,
                    "strategy": strategy,
                    "parameters": parameters,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "text_sha256": hashlib.sha256(
                        piece.encode("utf-8")
                    ).hexdigest(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )

            node = TextNode(
                id_=str(uuid5(NAMESPACE_URL, identity)),
                text=piece,
                metadata=metadata,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id=document.id_,
                    ),
                },
                # Служебные поля не добавляются к тексту эмбеддинга.
                # Они остаются доступны в metadata.
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
            )
            nodes.append(node)

    return nodes


def fixed_size(
    documents: list[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextNode]:
    """Baseline: TokenTextSplitter, без приоритета границ предложений."""
    _validate_size(chunk_size, chunk_overlap)

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenize,
    )

    return _make_nodes(
        documents,
        strategy="fixed",
        parameters={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        split=splitter.split_text,
    )


def recursive(
    documents: list[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextNode]:
    """SentenceSplitter: абзацы, русские предложения, мелкие части."""
    _validate_size(chunk_size, chunk_overlap)

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        chunking_tokenizer_fn=russian_sentences,
        tokenizer=tokenize,
    )

    return _make_nodes(
        documents,
        strategy="recursive",
        parameters={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "sentence_splitter": "razdel",
        },
        split=splitter.split_text,
    )


def semantic(
    documents: list[Document],
    *,
    embed_model: BaseEmbedding,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
    chunk_size: int | None = None,
    chunk_overlap: int = 64,
) -> list[TextNode]:
    """Semantic, опционально с ограничением длины длинных фрагментов.

    chunk_size=None:
        исходная semantic-стратегия.

    chunk_size задан:
        длинные semantic-фрагменты дополнительно разделяются
        через SentenceSplitter. Короткие сохраняются без изменений.
    """
    if buffer_size < 1:
        raise ValueError("buffer_size должен быть не меньше 1")

    if not 0 < breakpoint_percentile_threshold < 100:
        raise ValueError(
            "breakpoint_percentile_threshold должен быть между 0 и 100"
        )

    limiter = None

    if chunk_size is not None:
        _validate_size(chunk_size, chunk_overlap)

        limiter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
            chunking_tokenizer_fn=russian_sentences,
            tokenizer=tokenize,
        )

    splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        sentence_splitter=russian_sentences,
        include_metadata=False,
        include_prev_next_rel=False,
    )

    def split_one(text: str) -> list[str]:
        parts = splitter.get_nodes_from_documents(
            [Document(text=text)],
            show_progress=False,
        )

        result: list[str] = []

        for part in parts:
            piece = part.get_content(metadata_mode=MetadataMode.NONE)

            if limiter is not None and count_tokens(piece) > chunk_size:
                result.extend(limiter.split_text(piece))
            else:
                result.append(piece)

        return result

    parameters = {
        "buffer_size": buffer_size,
        "breakpoint_percentile_threshold": breakpoint_percentile_threshold,
        "embedding_model": embed_model.model_name,
        "sentence_splitter": "razdel",
    }

    strategy = "semantic"

    if chunk_size is not None:
        strategy = "semantic_recursive"
        parameters.update(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "post_splitter": "SentenceSplitter",
            }
        )

    return _make_nodes(
        documents,
        strategy=strategy,
        parameters=parameters,
        split=split_one,
    )


def whole_bug(documents: list[Document]) -> list[TextNode]:
    """Дополнительный вариант: одна карточка — один целый чанк."""
    return _make_nodes(
        documents,
        strategy="whole_bug",
        parameters={},
        split=lambda text: [text],
    )