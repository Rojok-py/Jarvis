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


# ── Создание и редактирование файлов ────────────────────────


def create_text_file(filename: str, content: str, directory: str = "data") -> Path:
    """
    Создать текстовый файл в указанной папке (data/ или out/).

    Parameters
    ----------
    filename : str
        Имя файла (например, ``notes.txt``).
    content : str
        Содержимое файла.
    directory : str
        ``"data"`` или ``"out"`` — целевая папка.

    Returns
    -------
    Path
        Путь к созданному файлу.
    """
    target_dir = OUT_DIR if directory == "out" else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(content, encoding="utf-8")
    log.info("Создан файл: %s", path)
    return path


def edit_text_file(
    filename: str,
    *,
    new_content: str | None = None,
    append: str | None = None,
    old_text: str | None = None,
    new_text: str | None = None,
) -> Path:
    """
    Редактировать текстовый файл в data/ или out/.

    Поддерживает три режима:
    1. **Полная перезапись** — ``new_content`` заменяет всё содержимое.
    2. **Дописать в конец** — ``append`` добавляется в конец файла.
    3. **Замена фрагмента** — ``old_text`` → ``new_text``.

    Parameters
    ----------
    filename : str
        Имя файла (ищется сначала в data/, затем в out/).
    new_content : str | None
        Полное новое содержимое (перезапись).
    append : str | None
        Текст для добавления в конец.
    old_text : str | None
        Фрагмент для замены.
    new_text : str | None
        Замена для ``old_text``.

    Returns
    -------
    Path
        Путь к изменённому файлу.

    Raises
    ------
    FileNotFoundError
        Если файл не найден ни в data/, ни в out/.
    ValueError
        Если ``old_text`` не найден в файле.
    """
    path = _find_file(filename)

    if new_content is not None:
        path.write_text(new_content, encoding="utf-8")
        log.info("Файл перезаписан: %s", path)
    elif append is not None:
        existing = path.read_text(encoding="utf-8")
        path.write_text(existing + append, encoding="utf-8")
        log.info("Файл дополнен: %s", path)
    elif old_text is not None and new_text is not None:
        existing = path.read_text(encoding="utf-8")
        if old_text not in existing:
            raise ValueError(f"Фрагмент «{old_text[:60]}…» не найден в {filename}")
        updated = existing.replace(old_text, new_text, 1)
        path.write_text(updated, encoding="utf-8")
        log.info("Файл отредактирован (замена): %s", path)
    else:
        raise ValueError("Укажите new_content, append, или пару old_text/new_text.")

    return path


def read_text_file(filename: str) -> tuple[Path, str]:
    """
    Прочитать текстовый файл из data/ или out/.

    Returns
    -------
    tuple[Path, str]
        (путь к файлу, содержимое).
    """
    path = _find_file(filename)
    content = path.read_text(encoding="utf-8")
    return path, content


def _find_file(filename: str) -> Path:
    """Найти файл в data/ или out/ по имени."""
    for d in (DATA_DIR, OUT_DIR):
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Файл «{filename}» не найден ни в data/, ни в out/."
    )
