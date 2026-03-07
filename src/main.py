"""
J.A.R.V.I.S. — точка входа.

Запуск: python -m src.main
или:   poetry run python src/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication

from src.core.engine import JarvisEngine
from src.ui.interface import JarvisWindow


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()
    log = logging.getLogger("jarvis")

    log.info("Инициализация J.A.R.V.I.S...")

    # Движок Gemini
    engine = JarvisEngine()

    # Запуск PyQt6
    app = QApplication(sys.argv)
    app.setApplicationName("J.A.R.V.I.S.")

    window = JarvisWindow(engine)
    window.show()

    log.info("J.A.R.V.I.S. запущен. Режим: текстовый.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
