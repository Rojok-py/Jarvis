"""
Audio Recorder — запись голоса через FFmpeg (PulseAudio).

Использует subprocess + ffmpeg -f pulse, никаких C-зависимостей.
Результат сохраняется в data/input.wav.
"""

from __future__ import annotations

import logging
import subprocess
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Корень проекта — два уровня вверх от этого файла
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "input.wav"


def _check_ffmpeg() -> str:
    """Возвращает путь к ffmpeg или кидает RuntimeError."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установите: sudo pacman -S ffmpeg"
        )
    return path


def record(duration: int = 5, output: Path | None = None) -> Path:
    """
    Записать аудио с микрофона через PulseAudio.

    Parameters
    ----------
    duration : int
        Длительность записи в секундах.
    output : Path | None
        Куда сохранить .wav файл. По умолчанию — data/input.wav.

    Returns
    -------
    Path
        Путь к записанному файлу.
    """
    ffmpeg = _check_ffmpeg()
    output = output or DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",                       # перезаписать без вопросов
        "-f", "pulse",              # PulseAudio как источник
        "-ac", "1",                 # моно (до -i для корректного захвата)
        "-ar", "16000",             # 16 kHz — оптимально для STT
        "-i", "default",            # устройство по умолчанию
        "-t", str(duration),        # продолжительность
        "-acodec", "pcm_s16le",     # 16-bit PCM Little-Endian
        "-af", "highpass=f=80,lowpass=f=8000,volume=2.0",  # фильтр шума + усиление
        str(output),
    ]

    log.info("Запись %d сек → %s", duration, output)
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=duration + 10)
    except subprocess.TimeoutExpired:
        log.error("FFmpeg завис при записи — timeout")
        raise
    except subprocess.CalledProcessError as exc:
        log.error("FFmpeg ошибка: %s", exc.stderr.decode(errors="replace"))
        raise

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Файл записи пуст или не создан: {output}")

    log.info("Запись завершена: %s (%d байт)", output, output.stat().st_size)
    return output
