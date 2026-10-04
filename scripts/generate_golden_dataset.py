"""Генерация сырого golden dataset через RAGAS TestsetGenerator.

Внимание: скрипт делает платные LLM/embedding-запросы.
После генерации CSV нужно вручную вычитать и сохранить финальный JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx
import pymupdf as fitz
from langchain_core.documents import Document
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue

        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                pages = []
                for page in pdf:
                    text = page.get_text().strip()
                    if text:
                        pages.append(text)
                text = "\n\n".join(pages).strip()
        else:
            text = path.read_text(encoding="utf-8-sig").strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "relative_path": path.relative_to(ROOT).as_posix(),
                },
            )
        )

    if len(documents) < 30:
        raise ValueError(f"Слишком мало документов для генерации: {len(documents)}")

    return documents


async def close_async(client) -> None:
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/training_bugs",
        help="Папка с учебным корпусом MD/PDF",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=40,
        help="Сколько примеров сгенерировать. Лучше 40, чтобы после вычитки осталось 30+.",
    )
    parser.add_argument(
        "--output",
        default="tests/eval/golden_dataset_raw.csv",
        help="Куда сохранить сырой CSV",
    )
    args = parser.parse_args()

    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator

    from app.evaluation.clients import EvalSettings, build_eval_judge, get_eval_connection

    data_dir = (ROOT / args.data_dir).resolve()
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    documents = read_documents(data_dir)

    evaluation = EvalSettings()
    llm = get_eval_connection()

    http = httpx.AsyncClient(
        proxy=llm.openai_proxy_url or None,
        timeout=evaluation.request_timeout,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=0,
        ),
    )

    client = AsyncOpenAI(
        api_key=llm.openai_api_key.get_secret_value(),
        base_url=llm.base_url,
        http_client=http,
        timeout=evaluation.request_timeout,
        max_retries=0,
    )

    try:
        judge = build_eval_judge(
            client,
            evaluation.judge_model,
        )
            
        embeddings = OpenAIEmbeddings(
            client=client,
            model=evaluation.embedding_model,
        )

        generator = TestsetGenerator(
            llm=judge,
            embedding_model=embeddings,
        )

        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=args.size,
        )

        frame = testset.to_pandas()
        frame.to_csv(output, index=False, encoding="utf-8-sig")

        print(f"Документов прочитано: {len(documents)}")
        print(f"Сырой golden dataset сохранён: {output}")

    finally:
        async def close_clients() -> None:
            await close_async(client)
            await close_async(http)

        asyncio.run(close_clients())


if __name__ == "__main__":
    main()