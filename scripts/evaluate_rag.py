"""Сравнение RAG через HTTP и bare-metal на пяти вопросах."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from app.core.rag_prompts import FALLBACK_ANSWER
    from app.services.rag_baremetal import BareMetalRAGService

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-url",
        required=True,
        help="Адрес работающего backend",
    )
    args = parser.parse_args()

    questions_path = PROJECT_ROOT / "tests" / "eval" / "rag_questions.json"

    with questions_path.open(encoding="utf-8-sig") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError("В файле должно быть ровно пять вопросов")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    report_dir = PROJECT_ROOT / "docs"
    report_dir.mkdir(exist_ok=True)

    # Отдельный файл для каждого прогона.
    report_path = report_dir / f"rag_evaluation_{stamp}.jsonl"

    baremetal = BareMetalRAGService()

    try:
        baremetal.build()

        with (
            httpx.Client(
                base_url=args.backend_url.rstrip("/") + "/",
                timeout=180,
                trust_env=False,
            ) as http,
            report_path.open("x", encoding="utf-8") as report,
        ):
            for case in questions:
                for implementation in ("llamaindex", "baremetal"):
                    print()
                    print(
                        f"Вопрос {case['id']} "
                        f"({case['kind']}), версия: {implementation}"
                    )
                    print(case["question"])

                    started = perf_counter()

                    try:
                        if implementation == "llamaindex":
                            response = http.post(
                                "rag/query",
                                json={"question": case["question"]},
                            )
                            response.raise_for_status()
                            result = response.json()
                        else:
                            result = baremetal.answer(case["question"])

                        elapsed = perf_counter() - started

                        if not isinstance(result.get("answer"), str):
                            raise ValueError("В ответе отсутствует строка answer")

                        sources = result.get("sources")

                        if not isinstance(sources, list):
                            raise ValueError("В ответе отсутствует список sources")

                        found_sources = {
                            source["source"]
                            for source in sources
                            if source.get("source")
                        }

                        expected = set(case["expected_sources"])
                        missing = sorted(expected - found_sources)

                        fallback = (
                            result["answer"].strip() == FALLBACK_ANSWER
                        )

                        # Это проверка retrieval, а не качества всего ответа.
                        retrieval_ok = (
                            expected.issubset(found_sources)
                            if expected
                            else None
                        )

                        row = {
                            "case_id": case["id"],
                            "kind": case["kind"],
                            "implementation": implementation,
                            "question": case["question"],
                            "expected_sources": case["expected_sources"],
                            "expected_answer": case["expected_answer"],
                            "result": result,
                            "top_1_source": sources[0] if sources else None,
                            "retrieval_ok": retrieval_ok,
                            "missing_sources": missing,
                            "fallback": fallback,
                            "sources_count": len(sources),
                            "elapsed_seconds": round(elapsed, 3),
                            "manual_assessment": None,
                            "hypothesis": None,
                        }

                        print(f"top_score: {result['top_score']}")
                        print(
                            "Источники:",
                            [source.get("source") for source in sources],
                        )

                        if expected:
                            print(
                                "Ожидаемые источники найдены:",
                                "ДА" if retrieval_ok else "НЕТ",
                            )
                            if missing:
                                print("Не найдены:", missing)
                        else:
                            print(
                                "Точный fallback:",
                                "ДА" if fallback else "НЕТ",
                            )

                        print(f"Количество источников: {len(sources)}")
                        print(f"Время всего ответа: {elapsed:.3f} сек.")
                        print("Ответ:")
                        print(result["answer"])

                    except Exception as exc:
                        row = {
                            "case_id": case["id"],
                            "implementation": implementation,
                            "question": case["question"],
                            "error_type": type(exc).__name__,
                        }

                        if isinstance(exc, httpx.HTTPStatusError):
                            row["http_status"] = exc.response.status_code

                        print("Ошибка:", row["error_type"])

                        if "http_status" in row:
                            print("HTTP status:", row["http_status"])

                        report.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                        report.flush()

                        # Не продолжаем платные вызовы после ошибки.
                        raise

                    report.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    report.flush()

    finally:
        baremetal.close()
        print()
        print(f"Файл результатов: {report_path}")


if __name__ == "__main__":
    main()