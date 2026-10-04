"""Проверка зависимостей RAG-оценки без вызовов моделей."""

from __future__ import annotations

import inspect
import os
from importlib.metadata import version


# Отключаем сбор телеметрии RAGAS для этой проверки.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"


def main() -> None:
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

    import pandas
    import phoenix

    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import discrete_metric
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.testset import TestsetGenerator

    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    from openinference.instrumentation.llama_index import (
        LlamaIndexInstrumentor,
    )
    from openinference.instrumentation.openai import (
        OpenAIInstrumentor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from phoenix.otel import register

    packages = [
        "ragas",
        "langchain-community",
        "langchain-core",
        "anthropic",
        "openai",
        "pandas",
        "arize-phoenix",
        "arize-phoenix-otel",
        "llama-index-core",
        "llama-index-vector-stores-qdrant",
        "openinference-instrumentation-llama-index",
        "openinference-instrumentation-openai",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp",
    ]

    print("ВЕРСИИ")
    for package in packages:
        print(f"{package}=={version(package)}")

    print("\nСИГНАТУРЫ МЕТРИК")
    for metric in (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    ):
        print(
            f"{metric.__name__}.ascore"
            f"{inspect.signature(metric.ascore)}"
        )

    print("\nВСЕ ИМПОРТЫ УСПЕШНЫ")
    print("Клиенты моделей не создавались.")
    print("Генерация, эмбеддинги и оценивание не запускались.")


if __name__ == "__main__":
    main()