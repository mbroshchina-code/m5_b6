import argparse
import ast
import csv
import json
from pathlib import Path


def parse_contexts(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    return [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for index, row in enumerate(reader, start=2):
            user_input = (row.get("user_input") or "").strip()
            reference = (row.get("reference") or "").strip()
            reference_contexts = parse_contexts(row.get("reference_contexts") or "")

            if not user_input or not reference or not reference_contexts:
                print(f"Пропущена строка {index}: пустой user_input/reference/reference_contexts")
                continue

            rows.append(
                {
                    "user_input": user_input,
                    "reference_contexts": reference_contexts,
                    "reference": reference,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    print(f"Готово: сохранено {len(rows)} примеров в {output_path}")


if __name__ == "__main__":
    main()