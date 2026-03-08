"""
TTS Speaker — озвучка ответов через edge-tts + mpv.

edge-tts генерирует mp3, mpv его проигрывает.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Голос по умолчанию — русский мужской (для русскоязычного общения)
DEFAULT_VOICE = "ru-RU-DmitryNeural"

# Текущий процесс mpv (для прерывания)
_current_mpv_proc: subprocess.Popen | None = None


def _check_mpv() -> str:
    path = shutil.which("mpv")
    if path is None:
        raise RuntimeError("mpv не найден. Установите: sudo pacman -S mpv")
    return path


async def _generate_tts(text: str, voice: str, output: Path) -> None:
    """Сгенерировать mp3 через edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output))


def _clean_for_tts(text: str) -> str:
    """Очистить текст от markdown и спецсимволов для озвучки."""
    # Убираем markdown-форматирование
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **жирный** → жирный
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *курсив* → курсив
    text = re.sub(r'`(.+?)`', r'\1', text)          # `код` → код
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # ### заголовки
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)  # - пункты списка
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)   # 1. нумерация
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # [ссылка](url) → ссылка
    text = re.sub(r'```[\s\S]*?```', '', text)        # блоки кода
    # Убираем эмоджи и спецсимволы
    text = re.sub(r'[📂📄📑🎤✕❌✅🔍💡⚠️🚀✨]+', '', text)
    text = re.sub(r'[•►▸▹→←↑↓]', '', text)
    text = re.sub(r'---+', '', text)
    # Убираем лишние пробелы и пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def stop() -> None:
    """Остановить текущее воспроизведение TTS (убить mpv)."""
    global _current_mpv_proc
    if _current_mpv_proc is not None and _current_mpv_proc.poll() is None:
        _current_mpv_proc.terminate()
        try:
            _current_mpv_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _current_mpv_proc.kill()
        log.info("TTS: воспроизведение остановлено")
    _current_mpv_proc = None


def speak(text: str, voice: str = DEFAULT_VOICE) -> None:
    """
    Озвучить текст: edge-tts → mp3 → mpv.

    Parameters
    ----------
    text : str
        Текст для произнесения.
    voice : str
        Имя голоса edge-tts.
    """
    text = _clean_for_tts(text)
    if not text.strip():
        return

    mpv = _check_mpv()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        log.info("TTS: генерация речи (%d символов)", len(text))
        asyncio.run(_generate_tts(text, voice, tmp_path))

        log.info("TTS: воспроизведение через mpv")
        global _current_mpv_proc
        _current_mpv_proc = subprocess.Popen(
            [mpv, "--no-video", "--really-quiet", str(tmp_path)],
        )
        try:
            _current_mpv_proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            _current_mpv_proc.kill()
        finally:
            _current_mpv_proc = None
    finally:
        tmp_path.unlink(missing_ok=True)
