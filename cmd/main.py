import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from usecase.pipe import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Пайплайн детекции, расчета метаданных и классификации зубов."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        help="Путь к папке с изображениями для обработки.",
    )
    return parser


def prompt_for_input_dir() -> Path:
    print("Приложение для детекции и классификации зубов.")
    print("Укажите путь к папке, в которой лежат изображения.")
    print("Пример: C:\\data\\teeth_images")
    user_input = input("Путь к папке: ").strip()
    if not user_input:
        raise ValueError("Путь к папке не был указан.")
    return Path(user_input)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_dir = args.input_dir if args.input_dir is not None else prompt_for_input_dir()

    try:
        results_dir = run_pipeline(input_dir)
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"Обработка завершена. Результаты сохранены в: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
