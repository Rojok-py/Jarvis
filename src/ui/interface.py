"""
Interface — главное окно J.A.R.V.I.S. на PyQt6.

Два режима: текстовый чат и голосовой ввод.
Футуристическая тёмная тема с фоновым изображением.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.animations import AnimationWidget

log = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


# ── Стиль ────────────────────────────────────────────────────

DARK_STYLE = """
QMainWindow {
    background-color: #0a0e17;
}
QWidget {
    background-color: #0a0e17;
    color: #c0d0e0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
QTextEdit {
    background-color: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
    color: #e0e8f0;
    selection-background-color: #1e3a5f;
}
QLineEdit {
    background-color: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
    color: #e0e8f0;
}
QLineEdit:focus {
    border-color: #00d4ff;
}
QPushButton {
    background-color: #1e3a5f;
    border: 1px solid #2a5a8f;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: bold;
    color: #00d4ff;
}
QPushButton:hover {
    background-color: #2a5a8f;
    border-color: #00d4ff;
}
QPushButton:pressed {
    background-color: #00d4ff;
    color: #0a0e17;
}
QPushButton#modeBtn {
    background-color: #0d2137;
    border: 2px solid #00d4ff;
    font-size: 12px;
    padding: 6px 14px;
}
QPushButton#voiceBtn {
    background-color: #1a0a2e;
    border: 2px solid #8b5cf6;
    color: #8b5cf6;
    font-size: 16px;
    min-width: 56px;
    min-height: 56px;
    border-radius: 28px;
}
QPushButton#voiceBtn:hover {
    background-color: #2d1a4e;
    border-color: #a78bfa;
}
QPushButton#voiceBtn:pressed {
    background-color: #8b5cf6;
    color: #0a0e17;
}
QPushButton#stopBtn {
    background-color: #2a0a0a;
    border: 2px solid #ef4444;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: bold;
    color: #ef4444;
}
QPushButton#stopBtn:hover {
    background-color: #ef4444;
    color: #0a0e17;
}
QPushButton#stopBtn:pressed {
    background-color: #b91c1c;
    color: #ffffff;
}
QLabel#title {
    color: #00d4ff;
    font-size: 22px;
    font-weight: bold;
}
QLabel#status {
    color: #4a6a8a;
    font-size: 11px;
}
QScrollArea {
    border: none;
}
"""


class Mode(Enum):
    TEXT = auto()
    VOICE = auto()


# ── Фоновый виджет ──────────────────────────────────────────

class BackgroundWidget(QWidget):
    """Центральный виджет с фоновым изображением из assets/jarvis.webp."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bg_pixmap: QPixmap | None = None

        bg_path = _ASSETS_DIR / "jarvis.webp"
        if bg_path.exists():
            self._bg_pixmap = QPixmap(str(bg_path))
            log.info("Фон загружен: %s", bg_path)
        else:
            log.warning("Фон не найден: %s", bg_path)

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            painter = QPainter(self)
            painter.setOpacity(0.07)
            # Рисуем по центру, масштабируя с сохранением пропорций
            scaled = self._bg_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
        super().paintEvent(event)


# ── Worker-потоки (чтобы UI не зависал) ─────────────────────

class TextWorker(QThread):
    """Отправляет текст в engine в фоне."""
    finished = pyqtSignal(str)   # ответ Gemini
    error = pyqtSignal(str)

    def __init__(self, engine, text: str) -> None:
        super().__init__()
        self._engine = engine
        self._text = text

    def run(self) -> None:
        try:
            # Сначала проверяем команды, потом обычный чат
            result = self._engine.process_voice_command(self._text)
            if result is None:
                result = self._engine.send_text(self._text)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class VoiceWorker(QThread):
    """Записывает голос → отправляет в engine → получает ответ."""
    status = pyqtSignal(str)          # статус для UI
    finished = pyqtSignal(str, str)   # (транскрипция, ответ)
    error = pyqtSignal(str)

    def __init__(self, engine, recorder_mod, speaker_mod, duration: int = 5) -> None:
        super().__init__()
        self._engine = engine
        self._recorder = recorder_mod
        self._speaker = speaker_mod
        self._duration = duration

    def run(self) -> None:
        try:
            self.status.emit("Запись...")
            audio_path = self._recorder.record(duration=self._duration)

            self.status.emit("Распознавание...")
            transcript, reply = self._engine.send_audio(audio_path)

            self.status.emit("Озвучивание...")
            self._speaker.speak(reply)

            self.finished.emit(transcript, reply)
        except Exception as exc:
            self.error.emit(str(exc))


class IndexWorker(QThread):
    """Индексирует файлы из data/ в фоне."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine) -> None:
        super().__init__()
        self._engine = engine

    def run(self) -> None:
        try:
            result = self._engine.index_documents(force=True)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Главное окно ─────────────────────────────────────────────

class JarvisWindow(QMainWindow):
    """Главное окно J.A.R.V.I.S."""

    def __init__(self, engine) -> None:
        super().__init__()
        self._engine = engine
        self._mode = Mode.TEXT
        self._worker: QThread | None = None

        self.setWindowTitle("J.A.R.V.I.S.")
        self.setMinimumSize(700, 600)
        self.setStyleSheet(DARK_STYLE)

        self._build_ui()
        self._connect_signals()

    # ── Построение интерфейса ────────────────────────────────

    def _build_ui(self) -> None:
        central = BackgroundWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ── Header ───────────────────────────────────────────
        header = QHBoxLayout()
        self._title = QLabel("J.A.R.V.I.S.")
        self._title.setObjectName("title")
        header.addWidget(self._title)

        header.addStretch()

        self._index_btn = QPushButton("📑 Индексация")
        self._index_btn.setObjectName("modeBtn")
        self._index_btn.setToolTip("Проиндексировать файлы из data/ для RAG-поиска")
        header.addWidget(self._index_btn)

        self._mode_btn = QPushButton("Режим: ТЕКСТ")
        self._mode_btn.setObjectName("modeBtn")
        header.addWidget(self._mode_btn)

        self._exit_btn = QPushButton("✕")
        self._exit_btn.setObjectName("modeBtn")
        self._exit_btn.setToolTip("Выход из J.A.R.V.I.S.")
        self._exit_btn.setFixedWidth(36)
        self._exit_btn.setStyleSheet(
            "QPushButton { background-color: #2a0a0a; border: 2px solid #ef4444; "
            "color: #ef4444; font-size: 14px; font-weight: bold; "
            "min-width: 36px; max-width: 36px; padding: 6px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #ef4444; color: #0a0e17; }"
        )
        header.addWidget(self._exit_btn)

        root.addLayout(header)

        # ── Анимация ─────────────────────────────────────────
        self._animation = AnimationWidget()
        self._animation.setFixedHeight(200)
        root.addWidget(self._animation, alignment=Qt.AlignmentFlag.AlignCenter)

        # ── Чат-лог ──────────────────────────────────────────
        self._chat_log = QTextEdit()
        self._chat_log.setReadOnly(True)
        self._chat_log.setPlaceholderText("Здесь будет диалог...")
        root.addWidget(self._chat_log, stretch=1)

        # ── Панель ввода (текст) ─────────────────────────────
        self._text_panel = QWidget()
        text_layout = QHBoxLayout(self._text_panel)
        text_layout.setContentsMargins(0, 0, 0, 0)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Введите сообщение...")
        text_layout.addWidget(self._input, stretch=1)

        self._send_btn = QPushButton("Отправить")
        text_layout.addWidget(self._send_btn)

        self._stop_btn = QPushButton("⏹ Стоп")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setToolTip("Остановить ответ J.A.R.V.I.S.")
        self._stop_btn.setVisible(False)
        text_layout.addWidget(self._stop_btn)

        root.addWidget(self._text_panel)

        # ── Панель ввода (голос) ─────────────────────────────
        self._voice_panel = QWidget()
        voice_layout = QVBoxLayout(self._voice_panel)
        voice_layout.setContentsMargins(0, 0, 0, 0)
        voice_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._voice_btn = QPushButton("🎤")
        self._voice_btn.setObjectName("voiceBtn")
        voice_layout.addWidget(self._voice_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._voice_stop_btn = QPushButton("⏹ Стоп")
        self._voice_stop_btn.setObjectName("stopBtn")
        self._voice_stop_btn.setToolTip("Остановить воспроизведение J.A.R.V.I.S.")
        self._voice_stop_btn.setVisible(False)
        voice_layout.addWidget(self._voice_stop_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._voice_panel.setVisible(False)
        root.addWidget(self._voice_panel)

        # ── Статус-бар ───────────────────────────────────────
        self._status = QLabel("Готов к работе")
        self._status.setObjectName("status")
        root.addWidget(self._status)

    # ── Сигналы ──────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._mode_btn.clicked.connect(self._toggle_mode)
        self._send_btn.clicked.connect(self._on_send_text)
        self._input.returnPressed.connect(self._on_send_text)
        self._voice_btn.clicked.connect(self._on_voice_record)
        self._index_btn.clicked.connect(self._on_index_files)
        self._exit_btn.clicked.connect(self._on_exit)
        self._stop_btn.clicked.connect(self._on_stop)
        self._voice_stop_btn.clicked.connect(self._on_stop)

    def _on_exit(self) -> None:
        """Выход из приложения."""
        QApplication.quit()

    def _on_stop(self) -> None:
        """Остановить текущий ответ и TTS."""
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.quit()
            self._worker.wait(300)
            if self._worker.isRunning():
                self._worker.terminate()
            log.info("Worker остановлен пользователем")

        # Останавливаем TTS если воспроизводится
        try:
            from src.audio import speaker
            speaker.stop()
        except Exception:
            pass

        self._append_message("J.A.R.V.I.S.", "*[Ответ остановлен]*")
        self._set_busy(False)

    # ── Переключение режимов ─────────────────────────────────

    def _toggle_mode(self) -> None:
        if self._mode == Mode.TEXT:
            self._mode = Mode.VOICE
            self._mode_btn.setText("Режим: ГОЛОС")
            self._text_panel.setVisible(False)
            self._voice_panel.setVisible(True)
        else:
            self._mode = Mode.TEXT
            self._mode_btn.setText("Режим: ТЕКСТ")
            self._text_panel.setVisible(True)
            self._voice_panel.setVisible(False)

        self._status.setText(f"Режим: {self._mode.name}")

    # ── Текстовый ввод ───────────────────────────────────────

    def _on_send_text(self) -> None:
        text = self._input.text().strip()
        if not text:
            return

        self._append_message("Вы", text)
        self._input.clear()
        self._set_busy(True)

        self._worker = TextWorker(self._engine, text)
        self._worker.finished.connect(self._on_text_reply)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_text_reply(self, reply: str) -> None:
        self._append_message("J.A.R.V.I.S.", reply)
        self._set_busy(False)

    # ── Голосовой ввод ───────────────────────────────────────

    def _on_voice_record(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        from src.audio import recorder, speaker

        self._set_busy(True)
        self._worker = VoiceWorker(self._engine, recorder, speaker, duration=5)
        self._worker.status.connect(lambda s: self._status.setText(s))
        self._worker.finished.connect(self._on_voice_reply)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_voice_reply(self, transcript: str, reply: str) -> None:
        if transcript:
            self._append_message("Вы (голос)", transcript)
        self._append_message("J.A.R.V.I.S.", reply)
        self._set_busy(False)

    # ── Индексация файлов ────────────────────────────────────

    def _on_index_files(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._set_busy(True)
        self._status.setText("Индексация файлов...")
        self._worker = IndexWorker(self._engine)
        self._worker.finished.connect(self._on_index_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_index_done(self, result: str) -> None:
        self._append_message("J.A.R.V.I.S.", result)
        self._set_busy(False)

    # ── Утилиты ──────────────────────────────────────────────

    def _on_error(self, msg: str) -> None:
        self._append_message("ОШИБКА", msg)
        self._set_busy(False)

    def _append_message(self, sender: str, text: str) -> None:
        color = "#00d4ff" if sender == "J.A.R.V.I.S." else "#8b5cf6"
        if sender == "ОШИБКА":
            color = "#ef4444"
        html = (
            f'<p style="margin:4px 0">'
            f'<span style="color:{color}; font-weight:bold">[{sender}]</span> '
            f'{text}</p>'
        )
        self._chat_log.append(html)

    def _set_busy(self, busy: bool) -> None:
        self._send_btn.setEnabled(not busy)
        self._send_btn.setVisible(not busy)
        self._stop_btn.setVisible(busy)
        self._voice_btn.setVisible(not busy)
        self._voice_btn.setEnabled(not busy)
        self._voice_stop_btn.setVisible(busy)
        self._input.setEnabled(not busy)
        self._animation.set_active(busy)
        if busy:
            self._status.setText("Обработка...")
        else:
            self._status.setText("Готов к работе")
