"""Запуск многоформатной индексации Б5.5."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.ingestion import IngestionService


def main() -> None:
    os.chdir(ROOT)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Папка или файл внутри data",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Только чтение файлов, без API, Qdrant и переименований",
    )
    args = parser.parse_args()

    service = IngestionService()
    result = service.run(
        input_path=args.path,
        check_only=args.check,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.check:
        print(
            f"{result['readable']} readable, "
            f"{result['failed']} failed"
        )
    else:
        print(
            f"{result['changed']} changed, "
            f"{result['unchanged']} unchanged, "
            f"{result['failed']} failed"
        )
        print(f"points_count={result['points_count']}")

    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()