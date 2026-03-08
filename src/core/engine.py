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

# «Создай файл notes.txt с содержимым ...» / «Напиши файл ...»
_RE_CREATE_FILE = re.compile(
    r"(?:создай|напиши|запиши|сделай|create|write)\s+(?:текстовый\s+)?файл\s+(?P<file>[\w.\-/]+)"
    r"(?:\s+(?:с\s+содержимым|с\s+текстом|with|содержимое|содержание|with\s+content))?\s*(?P<content>.*)",
    re.IGNORECASE | re.DOTALL,
)

# «Измени файл notes.txt ...» / «Отредактируй файл ...» / «Допиши в файл ...»
_RE_EDIT_FILE = re.compile(
    r"(?:измени|отредактируй|редактируй|поменяй|обнови|edit|modify|update)\s+(?:файл\s+)?(?P<file>[\w.\-/]+)"
    r"\s*(?P<instruction>.*)",
    re.IGNORECASE | re.DOTALL,
)

# «Допиши в файл notes.txt ...» / «Добавь в файл ...»
_RE_APPEND_FILE = re.compile(
    r"(?:допиши|добавь|append)\s+(?:в\s+)?(?:файл\s+)?(?P<file>[\w.\-/]+)\s+(?P<content>.+)",
    re.IGNORECASE | re.DOTALL,
)

# «Запиши/Сохрани [текст] в файл X» — явная запись содержимого без имени файла впереди
_RE_WRITE_FILE = re.compile(
    r"(?:запиши|сохрани|запишите|сохраните|write|save)\s+"
    r"(?:это|следующее|текст|слова)?\s*"
    r"(?:в\s+файл\s+|в\s+файле\s+|в\s+|to\s+file\s+|to\s+)(?P<file>[\w.\-/]+)"
    r"(?:\s*[:,]?\s*(?P<content>.+))?",
    re.IGNORECASE | re.DOTALL,
)

# «В моём файле» / «из файла» / «расскажи про файл(ы)» / «что в data» → RAG/анализ
_RE_FILE_QUERY = re.compile(
    r"(?:"
    r"в\s+(?:моем|моём|этом|нашем|данном|своём|своем)?\s*файл(?:е|ах)"
    r"|из\s+(?:моего|этого|своего)?\s*файл(?:а|ов)"
    r"|(?:прочитай|прочти|открой|читай)\s+(?:мой\s+|этот\s+|данный\s+)?файл"
    r"|покажи\s+содержимое\s+файла"
    r"|что\s+(?:есть|написано|содержит\s*ся)?\s*(?:в|во)\s+(?:моем|моём|этом)?\s*файл(?:е|ах)"
    r"|расскажи\s+(?:про|о|об)\s+(?:файл(?:е|ах|ы|ов)?)"
    r"|(?:что|чего|какие)\s+(?:в|за)\s+(?:файл(?:ах|е|ы|ов)?|папке\s+data|моих\s+файлах)"
    r"|(?:у\s+тебя|у\s+тебя\s+есть|в\s+data)\s+(?:файл(?:ы|ов|ах)?|есть\s+файл(?:ы|ов)?)"    r"|у\s+тебя\s+в\s+папке\s+data"
    r"|есть\s+файл(?:ы|ов)?\s+расскажи"    r"|(?:что|какие)\s+у\s+(?:тебя|меня)\s+файл(?:ы|ов|ах)?"
    r"|про\s+(?:мои|свои|эти|все)\s+файл(?:ы|ах|ов)?"
    r")",
    re.IGNORECASE,
)

# «Агент / агенты / исследуй с агентами / dual agent» — запуск двухагентного пайплайна
_RE_AGENTS = re.compile(
    r"(?:запусти\s+агент(?:ов|а)?|используй\s+агент(?:ов|а)?|исследуй\s+с\s+агентами?"
    r"|dual.?agent|агентный\s+поиск|через\s+агент(?:ов|а)?|ответь\s+через\s+агент(?:ов|а)?)"
    r"(?:\s+(?:про\s+|о\s+|по\s+|на\s+тему\s+))?(?P<query>.*)",
    re.IGNORECASE | re.DOTALL,
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

    def _search_complete(self, query: str, prompt: str) -> str:
        """
        Обработать результаты поиска через Groq без добавления в историю чата.
        Добавляет в историю только финальный краткий ответ (без сырых данных поиска).
        """
        response = _retry_on_429(
            self._client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self._messages[1:],   # история без system
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content.strip()

        # В историю добавляем только пользовательский запрос + краткий ответ
        self._messages.append({"role": "user", "content": query})
        self._messages.append({"role": "assistant", "content": reply})

        if len(self._messages) > 41:
            self._messages = [self._messages[0]] + self._messages[-40:]

        log.info("Search → Groq: %s", reply[:80])
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
            if not results or "не удалось" in results.lower() or results.strip() == "Результатов не найдено.":
                return self._chat_complete(
                    f"Пользователь спросил: «{text}». "
                    "Поиск не дал результатов. Скажи об этом честно и предложи альтернативы."
                )
            summary_prompt = (
                f"Ниже приведены РЕАЛЬНЫЕ актуальные результаты поиска DuckDuckGo "
                f"по запросу «{query}».\n\n"
                f"{results}\n\n"
                "Используй эти данные как единственный источник информации. "
                "Извлеки ключевые факты и дай чёткий, структурированный ответ. "
                "Если это погода — укажи температуру, осадки, условия. "
                "Если это новости — кратко изложи суть. "
                "Указывай источники (URL). Отвечай на русском языке."
            )
            # Прямой вызов без добавления в историю чата
            return self._search_complete(query, summary_prompt)

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

        # ── Агентный поиск (LangChain dual-agent) ──
        m = _RE_AGENTS.search(text)
        if m:
            query = (m.group("query") or "").strip().rstrip(".,;:!?")
            if not query:
                query = text
            log.info("Команда: dual-agent '%s'", query)
            return self._handle_dual_agent(query)

        # ── Создание файла ──
        m = _RE_CREATE_FILE.search(text)
        if m:
            filename = m.group("file").strip().rstrip(".,;:!?")
            raw_content = (m.group("content") or "").strip().rstrip(".,;:!?")
            log.info("Команда: создать файл '%s'", filename)
            return self._handle_create_file(filename, raw_content, text)

        # ── Запись текста в файл (явная: «запиши X в файл Y») ──
        m = _RE_WRITE_FILE.search(text)
        if m:
            filename = m.group("file").strip().rstrip(".,;:!?")
            raw_content = (m.group("content") or "").strip()
            log.info("Команда: запись в файл '%s'", filename)
            return self._handle_write_to_file(filename, raw_content, text)

        # ── Дописать в файл ──
        m = _RE_APPEND_FILE.search(text)
        if m:
            filename = m.group("file").strip().rstrip(".,;:!?")
            raw_content = (m.group("content") or "").strip()
            log.info("Команда: дописать в файл '%s'", filename)
            return self._handle_append_file(filename, raw_content, text)

        # ── Редактирование файла ──
        m = _RE_EDIT_FILE.search(text)
        if m:
            filename = m.group("file").strip().rstrip(".,;:!?")
            instruction = (m.group("instruction") or "").strip()
            log.info("Команда: редактировать файл '%s'", filename)
            return self._handle_edit_file(filename, instruction, text)

        # ── Обращение к файлам через естественный язык ──────────────────────
        if _RE_FILE_QUERY.search(text):
            log.info("Команда: запрос к файлам (natural language) → RAG")
            files = self._get_all_data_files()
            if files:
                return self.rag_query(text)
            return (
                "В папке data/ пока нет файлов, сэр. Добавьте документы "
                "и используйте «📑 Индексация» или скажите «индексируй файлы»."
            )

        # ── Упоминание конкретного имени файла из data/ («расскажи про Algoritm.pdf») ──
        file_hit = self._detect_filename_in_text(text)
        if file_hit:
            log.info("Команда: упоминание файла '%s' → RAG", file_hit.name)
            if not self._rag.is_indexed:
                self.index_documents()
            return self.rag_query(text)

        return None

    # ── Создание и редактирование файлов ────────────────────────

    def _detect_filename_in_text(self, text: str) -> "Path | None":
        """
        Проверить, упоминается ли в тексте имя или часть имени файла из data/.
        Возвращает Path если найдено совпадение, иначе None.
        """
        files = self._get_all_data_files()
        if not files:
            return None
        text_lower = text.lower()
        for f in files:
            # Полное имя файла
            if f.name.lower() in text_lower:
                return f
            # Без расширения
            if f.stem.lower() in text_lower and len(f.stem) > 3:
                return f
            # Слова из имени (для «26.05.07 method 001» → «method 001» или «method»)
            stem_words = re.split(r'[\s._\-]+', f.stem.lower())
            meaningful = [w for w in stem_words if len(w) > 3]
            if meaningful and all(w in text_lower for w in meaningful):
                return f
        return None

    def _handle_create_file(
        self, filename: str, raw_content: str, full_text: str
    ) -> str:
        """Создать текстовый файл.  LLM генерирует / расширяет содержимое во всех случаях."""
        from src.tools.file_ops import create_text_file

        content = self._generate_file_content(filename, full_text, hint=raw_content)
        path = create_text_file(filename, content)
        return (
            f"✅ Файл **{filename}** создан в `{path.parent.name}/`.\n\n"
            f"Содержимое ({len(content)} символов):\n```\n{content[:500]}"
            + ("\n… [обрезано]" if len(content) > 500 else "")
            + "\n```"
        )

    def _handle_append_file(
        self, filename: str, raw_content: str, full_text: str
    ) -> str:
        """Дописать текст в конец существующего файла. LLM расширяет введённую идею."""
        from src.tools.file_ops import edit_text_file

        content = self._generate_file_content(filename, full_text, hint=raw_content)

        try:
            path = edit_text_file(filename, append="\n" + content)
        except FileNotFoundError:
            from src.tools.file_ops import create_text_file
            path = create_text_file(filename, content)
            return (
                f"Файл **{filename}** не существовал — создал новый в "
                f"`{path.parent.name}/` и записал содержимое."
            )
        return f"✅ Дописал в **{filename}** ({path.parent.name}/):\n```\n{content[:300]}```"

    def _handle_edit_file(
        self, filename: str, instruction: str, full_text: str
    ) -> str:
        """Отредактировать файл — LLM генерирует новое содержимое на основе текущего."""
        from src.tools.file_ops import edit_text_file, read_text_file

        try:
            path, current = read_text_file(filename)
        except FileNotFoundError:
            return (
                f"Файл **{filename}** не найден ни в data/, ни в out/. "
                "Может, создать его? Скажите: «создай файл {filename} …»"
            )

        # Ограничим длину текущего содержимого для контекста
        truncated = current[:30000]
        if len(current) > 30000:
            truncated += "\n\n... [текст обрезан]"

        prompt = (
            f"Текущее содержимое файла «{filename}»:\n\n"
            f"```\n{truncated}\n```\n\n"
            f"Пользователь просит: {full_text}\n\n"
            "Верни ТОЛЬКО новое полное содержимое файла — без объяснений, "
            "без обёрток в markdown-блоки. Только текст файла."
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
        new_content = response.choices[0].message.content.strip()

        # Убираем markdown-обёртку если LLM всё же добавил
        new_content = self._strip_markdown_wrapper(new_content)

        edit_text_file(filename, new_content=new_content)
        return (
            f"✅ Файл **{filename}** обновлён.\n\n"
            f"Новое содержимое ({len(new_content)} символов):\n```\n{new_content[:500]}"
            + ("\n… [обрезано]" if len(new_content) > 500 else "")
            + "\n```"
        )

    def _handle_write_to_file(
        self, filename: str, raw_content: str, full_text: str
    ) -> str:
        """Записать (создать или перезаписать) файл.

        Всегда генерирует содержимое через LLM:
        - если пользователь дал текст/тему — LLM расширяет это в полноценную запись
        - если ничего нет — LLM генерирует с нуля по имени файла и полному запросу
        """
        from src.tools.file_ops import create_text_file, edit_text_file

        if raw_content:
            # Просим LLM расширить/улучшить введённый пользователем текст
            prompt = (
                f"Пользователь хочет записать в файл «{filename}» следующее:\n"
                f"{raw_content}\n\n"
                "Напиши полноценное, законченное содержимое файла на основе этого текста. "
                "Сохрани смысл, но оформи красиво и структурировано. "
                "Верни ТОЛЬКО текст файла — без объяснений и без markdown-обёрток."
            )
        else:
            prompt = (
                f"Пользователь просит записать что-то в файл «{filename}».\n"
                f"Полный запрос: {full_text}\n\n"
                "Сгенерируй подходящее содержимое файла. "
                "Верни ТОЛЬКО текст файла — без объяснений и без markdown-обёрток."
            )

        response = _retry_on_429(
            self._client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2048,
        )
        content = self._strip_markdown_wrapper(
            response.choices[0].message.content.strip()
        )

        try:
            path = edit_text_file(filename, new_content=content)
            action = "Обновлён"
        except FileNotFoundError:
            path = create_text_file(filename, content)
            action = "Создан"

        log.info("%s файл: %s", action, path)
        return (
            f"✅ {action} файл **{filename}** в `{path.parent.name}/`.\n\n"
            f"Содержимое ({len(content)} символов):\n"
            f"```\n{content[:500]}"
            + ("\n… [обрезано]" if len(content) > 500 else "")
            + "\n```"
        )

    def _handle_dual_agent(self, query: str) -> str:
        """Запустить LangChain dual-agent (Research + Editor) и вернуть ответ."""
        try:
            from src.core.agents import run_dual_agent

            log.info("Запуск dual-agent pipeline для: '%s'", query[:80])
            result = run_dual_agent(query)

            search_note = ""
            if result["searched"]:
                search_note = (
                    "\n\n*🔍 Выполнен автоматический веб-поиск DuckDuckGo*"
                )

            return (
                f"**[Research Agent → Editor Agent]**\n\n"
                f"{result['final']}"
                f"{search_note}"
            )
        except Exception as exc:
            log.error("Ошибка dual-agent: %s", exc)
            # Фолбэк на обычный Groq чат
            return self.send_text(query)

    def _generate_file_content(
        self, filename: str, user_request: str, hint: str = ""
    ) -> str:
        """Попросить LLM сгенерировать / расширить содержимое файла.

        hint : str
            Если пользователь дал текст/тему — LLM расширяет его в полноценную запись.
        """
        if hint:
            prompt = (
                f"Пользователь хочет записать в файл «{filename}» следующее:\n{hint}\n\n"
                "Напиши полноценное, законченное содержимое файла на основе этой темы или текста. "
                "Сохрани смысл, но оформи красиво и структурированно. "
                "Верни ТОЛЬКО текст файла — без объяснений и без markdown-обёрток."
            )
        else:
            prompt = (
                f"Пользователь просит создать файл «{filename}».\n"
                f"Полный запрос: {user_request}\n\n"
                "Сгенерируй подходящее содержимое файла. "
                "Верни ТОЛЬКО текст файла — без объяснений и без markdown-обёрток."
            )
        response = _retry_on_429(
            self._client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
        content = response.choices[0].message.content.strip()
        return self._strip_markdown_wrapper(content)

    @staticmethod
    def _strip_markdown_wrapper(text: str) -> str:
        """Убрать ```…``` обёртку если LLM добавил."""
        if text.startswith("```") and text.endswith("```"):
            lines = text.split("\n")
            # Убираем первую строку (```lang) и последнюю (```)
            return "\n".join(lines[1:-1]).strip()
        return text

    def reset_chat(self) -> None:
        """Сбросить историю диалога."""
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        log.info("История чата сброшена")
