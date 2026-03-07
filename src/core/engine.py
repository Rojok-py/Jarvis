"""
Groq Engine — ядро J.A.R.V.I.S.

Использует Groq SDK (OpenAI-совместимый API).

- Текстовый чат (send_text) — с ручной историей сообщений
- Голосовой ввод: Whisper STT → чат (send_audio)
- Анализ файлов (analyze_file) — чтение содержимого + чат
- Маршрутизация голосовых команд (process_voice_command)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from groq import Groq, RateLimitError
from dotenv import load_dotenv
import os

from src.core.rag import RAGIndex

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
OUT_DIR = _PROJECT_ROOT / "out"

MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"

# ── Retry при 429 (rate limit) ──────────────────────────────

_MAX_RETRIES = 3
_RETRY_DELAY = 5  # секунд


def _retry_on_429(func, *args, **kwargs):
    """Вызывает func с повторами при 429 Too Many Requests."""
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAY * (attempt + 1)
                log.warning("429 Rate limit — жду %d сек (попытка %d/%d)", wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait)
            else:
                raise
    return None  # unreachable


# Системный промпт — характер J.A.R.V.I.S.
SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., an advanced AI assistant created by Tony Stark. "
    "You are polite, efficient, and slightly witty with a British accent. "
    "You speak in a technical but accessible manner. "
    "Keep responses concise unless asked for detail. "
    "If the user speaks Russian, answer in Russian while maintaining your character. "
    "Always address the user respectfully."
)

# ── Регулярные выражения для голосовых команд ────────────────

# «Проанализируй файл report.pdf и сохрани в summary.txt» (файл опционален)
_RE_ANALYZE = re.compile(
    r"(?:проанализируй|анализируй|анализ|analyze)\s*(?:файл(?:ы)?\s*)?(?P<file>[\w.\-/]*)"
    r"(?:\s+(?:и\s+)?(?:сохрани|save)\s+(?:в|as|to)\s+(?P<save>[\w.\-/]+))?",
    re.IGNORECASE,
)

# «Найди в интернете ...» / «Поищи ...» / «Search for ...»
_RE_SEARCH = re.compile(
    r"(?:найди\s+(?:в\s+интернете\s+)?(?:информацию\s+)?(?:про\s+|о\s+)?|поищи\s+(?:в\s+интернете\s+)?(?:информацию\s+)?(?:про\s+|о\s+)?|search\s+(?:for\s+)?|загугли\s+|гугли\s+|поиск\s+)(?P<query>.+)",
    re.IGNORECASE,
)

# «Покажи файлы» / «Список файлов» / «List files»
_RE_LIST_FILES = re.compile(
    r"(?:покажи|список|list)\s+(?:файл(?:ы|ов)?|files)",
    re.IGNORECASE,
)

# «Сброс чата» / «Новый чат» / «Reset»
_RE_RESET = re.compile(
    r"(?:сброс(?:\s+чата)?|новый\s+чат|очисти\s+чат|reset|clear\s+chat)",
    re.IGNORECASE,
)

# «Индексируй файлы» / «Обнови индекс» / «Index files»
_RE_INDEX = re.compile(
    r"(?:индекс(?:ируй|ация)|обнови\s+индекс|переиндекс|index\s+files?|reindex)",
    re.IGNORECASE,
)

# «Спроси по документам / файлам» / «RAG ...» / «По документам ...»
_RE_RAG_QUERY = re.compile(
    r"(?:по\s+документ(?:ам|у)|по\s+файл(?:ам|у)|rag|из\s+документ(?:ов|а)|в\s+документ(?:ах|е))\s*(?P<query>.+)",
    re.IGNORECASE,
)


class JarvisEngine:
    """Обёртка над Groq API для J.A.R.V.I.S."""

    def __init__(self) -> None:
        load_dotenv(_PROJECT_ROOT / ".env")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY не задан. Добавьте его в .env файл."
            )

        self._client = Groq(api_key=api_key)

        # Ручная история сообщений (Groq — stateless API)
        self._messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # RAG-индекс (sentence-transformers, без внешнего клиента)
        self._rag = RAGIndex()
        log.info("JarvisEngine инициализирован (%s, Groq SDK)", MODEL)

    # ── RAG ─────────────────────────────────────────────────

    @property
    def rag(self) -> RAGIndex:
        """Доступ к RAG-индексу."""
        return self._rag

    def index_documents(self, force: bool = False) -> str:
        """Проиндексировать файлы из data/."""
        try:
            count = self._rag.index_files(force=force)
            stats = self._rag.stats
            if count > 0:
                return (
                    f"Индексация завершена: обработано файлов — {count}, "
                    f"всего чанков — {stats['chunks']}."
                )
            return (
                f"Все файлы актуальны. В индексе: "
                f"{stats['files']} файлов, {stats['chunks']} чанков."
            )
        except Exception as exc:
            log.error("Ошибка индексации: %s", exc)
            return f"Ошибка при индексации: {exc}"

    def rag_query(self, query: str) -> str:
        """Запрос с RAG-контекстом."""
        if not self._rag.is_indexed:
            self.index_documents()

        context = self._rag.build_context(query)
        if not context:
            return self.send_text(query)

        augmented = (
            f"{context}\n\n"
            f"Вопрос пользователя: {query}\n\n"
            "Ответь на вопрос, используя фрагменты из документов выше. "
            "Если информации в документах недостаточно, скажи об этом."
        )
        log.info("RAG-запрос: %s", query[:80])
        return self._chat_complete(augmented)

    # ── Текстовый режим ──────────────────────────────────────

    def send_text(self, user_text: str) -> str:
        """Отправить текст и получить ответ."""
        log.info("Text → Groq: %s", user_text[:80])
        return self._chat_complete(user_text)

    def _chat_complete(self, user_text: str) -> str:
        """Отправить сообщение через Groq Chat Completions с историей."""
        self._messages.append({"role": "user", "content": user_text})

        response = _retry_on_429(
            self._client.chat.completions.create,
            model=MODEL,
            messages=self._messages,
            temperature=0.7,
            max_tokens=4096,
        )
        reply = response.choices[0].message.content.strip()

        self._messages.append({"role": "assistant", "content": reply})

        # Ограничиваем историю (system + последние 40 сообщений)
        if len(self._messages) > 41:
            self._messages = [self._messages[0]] + self._messages[-40:]

        log.info("Groq → Text: %s", reply[:80])
        return reply

    # ── Голосовой режим ──────────────────────────────────────

    def transcribe(self, audio_path: Path | None = None) -> str:
        """
        Транскрибировать аудио через Groq Whisper (только STT, без чата).

        Returns
        -------
        str
            Распознанный текст.
        """
        audio_path = audio_path or DATA_DIR / "input.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        log.info("Audio → Groq Whisper: %s", audio_path)

        with open(audio_path, "rb") as f:
            transcription = _retry_on_429(
                self._client.audio.transcriptions.create,
                model=WHISPER_MODEL,
                file=f,
                language="ru",
            )
        transcript = transcription.text.strip()
        log.info("Транскрипция: %s", transcript[:80])
        return transcript

    def send_audio(self, audio_path: Path | None = None) -> tuple[str, str]:
        """
        Транскрибировать аудио и получить ответ.

        Сначала проверяет, является ли транскрипция командой.
        Если нет — отправляет как обычное сообщение в чат.

        Returns
        -------
        tuple[str, str]
            (транскрипция, ответ JARVIS).
        """
        transcript = self.transcribe(audio_path)

        if not transcript:
            return "", "Я не расслышал. Повторите, пожалуйста."

        # Сначала проверяем команды (без лишнего чат-вызова)
        command_result = self.process_voice_command(transcript)
        if command_result is not None:
            log.info("Голосовая команда → ответ: %s", command_result[:80])
            return transcript, command_result

        # Обычный разговор → чат
        reply = self._chat_complete(transcript)
        log.info("Ответ: %s", reply[:80])
        return transcript, reply

    # ── Анализ файлов ────────────────────────────────────────

    def _find_file_in_data(self, hint: str) -> Path | None:
        """Найти файл в data/ по неточному имени (fuzzy match)."""
        if not DATA_DIR.exists():
            return None

        hint_clean = hint.strip().rstrip(".,;:!?")
        files = [
            f for f in DATA_DIR.iterdir()
            if f.is_file() and not f.name.startswith(".")
            and f.name != "input.wav"
        ]

        if not files:
            return None

        for f in files:
            if f.name == hint_clean:
                return f

        hint_lower = hint_clean.lower()
        for f in files:
            if f.name.lower() == hint_lower:
                return f

        for f in files:
            stem = f.stem.lower()
            if stem in hint_lower or hint_lower in stem:
                return f

        return None

    def _get_all_data_files(self) -> list[Path]:
        """Все файлы в data/ (кроме input.wav и скрытых)."""
        if not DATA_DIR.exists():
            return []
        return sorted(
            f for f in DATA_DIR.iterdir()
            if f.is_file() and not f.name.startswith(".")
            and f.name != "input.wav"
        )

    def analyze_file(self, filename: str | None = None, save_as: str | None = None) -> str:
        """Проанализировать файл из data/."""
        if not filename or not filename.strip() or filename.strip().rstrip(".,;:!?") == "":
            return self._analyze_all_files(save_as=save_as)

        file_path = self._find_file_in_data(filename)
        if file_path is None:
            log.info("Файл '%s' не найден, анализирую все файлы в data/", filename)
            return self._analyze_all_files(save_as=save_as)

        return self._analyze_single_file(file_path, save_as=save_as)

    def _read_file_content(self, file_path: Path) -> str:
        """Прочитать содержимое файла для анализа."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            import pymupdf
            text_parts: list[str] = []
            with pymupdf.open(str(file_path)) as doc:
                for page in doc:
                    text_parts.append(page.get_text())
            return "\n".join(text_parts)
        else:
            return file_path.read_text(encoding="utf-8", errors="replace")

    def _analyze_single_file(self, file_path: Path, save_as: str | None = None) -> str:
        """Анализ одного файла."""
        log.info("Анализ файла: %s", file_path)

        content = self._read_file_content(file_path)
        if not content.strip():
            return f"Файл {file_path.name} пуст."

        # Обрезаем слишком длинные файлы (лимит контекста)
        max_chars = 60000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... [текст обрезан]"

        prompt = (
            f"Содержимое файла «{file_path.name}»:\n\n"
            f"```\n{content}\n```\n\n"
            "Analyze this file thoroughly. Provide a structured summary "
            "with key points, main topics, and important details. "
            "If it's code, describe its purpose and structure. "
            "Answer in the same language as the document."
        )
        response = _retry_on_429(
            self._client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        summary = response.choices[0].message.content.strip()

        if save_as:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUT_DIR / save_as
            out_path.write_text(summary, encoding="utf-8")
            log.info("Результат сохранён: %s", out_path)

        return summary

    def _analyze_all_files(self, save_as: str | None = None) -> str:
        """Анализ всех файлов в data/."""
        files = self._get_all_data_files()
        if not files:
            return "В папке data/ нет файлов для анализа."

        log.info("Анализ всех файлов: %s", [f.name for f in files])
        results: list[str] = []

        for fp in files:
            try:
                summary = self._analyze_single_file(fp)
                results.append(f"📄 **{fp.name}**\n{summary}")
            except Exception as exc:
                results.append(f"📄 **{fp.name}** — ошибка: {exc}")
                log.error("Ошибка анализа %s: %s", fp.name, exc)

        full = "\n\n---\n\n".join(results)

        if save_as:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUT_DIR / save_as
            out_path.write_text(full, encoding="utf-8")
            log.info("Все результаты сохранены: %s", out_path)

        return full

    # ── Маршрутизация голосовых команд ───────────────────────

    def process_voice_command(self, transcript: str) -> str | None:
        """
        Распознать интент из транскрипции и выполнить команду.
        Возвращает None если это не команда (обычный разговор).
        """
        text = transcript.strip()
        if not text:
            return "Я не расслышал. Повторите, пожалуйста."

        m = _RE_ANALYZE.search(text)
        if m:
            filename = m.group("file").strip().rstrip(".,;:!?") if m.group("file") else ""
            save_as = m.group("save")
            log.info("Команда: анализ файла '%s' (сохранить: %s)", filename, save_as)
            return self.analyze_file(filename or None, save_as=save_as)

        m = _RE_SEARCH.search(text)
        if m:
            query = m.group("query").strip().rstrip(".,;:!?")
            log.info("Команда: веб-поиск '%s'", query)
            from src.tools.search import web_search
            results = web_search(query)
            summary_prompt = (
                f"Вот результаты поиска DuckDuckGo по запросу «{query}»:\n\n"
                f"{results}\n\n"
                "Дай краткий и полезный ответ на основе этих результатов. "
                "Указывай источники."
            )
            return self.send_text(summary_prompt)

        if _RE_LIST_FILES.search(text):
            log.info("Команда: список файлов")
            from src.tools.file_ops import list_data_files, list_output_files
            data_files = list_data_files()
            out_files = list_output_files()
            msg = "📂 **Файлы в data/:**\n"
            msg += "\n".join(f"  • {f}" for f in data_files) if data_files else "  (пусто)"
            msg += "\n\n📂 **Файлы в out/:**\n"
            msg += "\n".join(f"  • {f}" for f in out_files) if out_files else "  (пусто)"
            return msg

        if _RE_RESET.search(text):
            log.info("Команда: сброс чата")
            self.reset_chat()
            return "История чата очищена. Начнём с чистого листа, сэр."

        if _RE_INDEX.search(text):
            log.info("Команда: индексация файлов")
            return self.index_documents(force=True)

        m = _RE_RAG_QUERY.search(text)
        if m:
            query = m.group("query").strip()
            log.info("Команда: RAG-запрос '%s'", query)
            return self.rag_query(query)

        return None

    def reset_chat(self) -> None:
        """Сбросить историю диалога."""
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        log.info("История чата сброшена")
