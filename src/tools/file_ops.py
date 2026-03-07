"""
File Operations — утилиты для работы с файлами data/ и out/.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
OUT_DIR = _PROJECT_ROOT / "out"


def list_data_files() -> list[str]:
    """Список файлов в data/ (без скрытых)."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        f.name for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")
    )


def list_output_files() -> list[str]:
    """Список файлов в out/."""
    if not OUT_DIR.exists():
        return []
    return sorted(
        f.name for f in OUT_DIR.iterdir() if f.is_file() and not f.name.startswith(".")
    )


def read_data_file(filename: str) -> str:
    """Прочитать текстовый файл из data/."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return path.read_text(encoding="utf-8")


def save_to_output(filename: str, content: str) -> Path:
    """Сохранить текст в out/<filename>."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    log.info("Сохранено: %s", path)
    return path
