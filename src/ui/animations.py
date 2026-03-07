"""
Animations — виджет анимации (GIF) для центра UI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout

log = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


class AnimationWidget(QWidget):
    """
    Виджет с анимированным GIF в центре.

    Ищет файл assets/jarvis.gif — если нет, показывает текст-заглушку.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._movie: QMovie | None = None
        gif_path = _ASSETS_DIR / "jarvis.gif"

        if gif_path.exists():
            self._movie = QMovie(str(gif_path))
            self._movie.setScaledSize(QSize(280, 280))
            self._label.setMovie(self._movie)
            self._movie.start()
            log.info("Анимация загружена: %s", gif_path)
        else:
            self._label.setText("J.A.R.V.I.S.")
            self._label.setStyleSheet(
                "color: #00d4ff; font-size: 36px; font-weight: bold;"
            )
            log.warning("GIF не найден: %s — показываем текст", gif_path)

        self._layout.addWidget(self._label)

    def set_active(self, active: bool) -> None:
        """Запустить/остановить анимацию."""
        if self._movie is None:
            return
        if active:
            self._movie.start()
        else:
            self._movie.stop()
