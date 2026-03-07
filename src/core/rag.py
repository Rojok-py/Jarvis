"""
RAG (Retrieval-Augmented Generation) — индексация и поиск по файлам.

Модуль читает файлы из data/, разбивает на чанки, строит
TF-IDF векторы через scikit-learn и при запросе находит
наиболее релевантные фрагменты для обогащения контекста LLM.

Поддерживаемые форматы: PDF, TXT, MD, CSV, JSON, PY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
_CACHE_DIR = _PROJECT_ROOT / ".rag_cache"

# Параметры чанкинга
CHUNK_SIZE = 800       # символов
CHUNK_OVERLAP = 200    # перекрытие

# Топ-K фрагментов для контекста
TOP_K = 5

# Расширения файлов для индексации
_SUPPORTED_EXT = {".pdf", ".txt", ".md", ".csv", ".json", ".py"}


# ── Парсинг файлов ──────────────────────────────────────────

def _extract_text_pdf(path: Path) -> str:
    """Извлечь текст из PDF через PyMuPDF."""
    import pymupdf

    text_parts: list[str] = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_text(path: Path) -> str:
    """Извлечь текст из файла по расширению."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_text_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


# ── Чанкинг ─────────────────────────────────────────────────

def _split_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                  overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Разбить текст на перекрывающиеся чанки."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ── Утилиты ─────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """SHA-256 хеш файла для кеширования."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ── Основной класс RAG ──────────────────────────────────────

class RAGIndex:
    """
    Индекс для Retrieval-Augmented Generation.

    Индексирует файлы из data/, строит TF-IDF матрицу,
    кеширует результаты и выполняет семантический поиск.
    """

    def __init__(self) -> None:
        self._chunks: list[dict] = []       # {"file": str, "text": str}
        self._tfidf_matrix = None           # scipy sparse matrix
        self._vectorizer: TfidfVectorizer | None = None
        self._indexed_files: dict[str, str] = {}  # filename → hash

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_path = _CACHE_DIR / "index.json"
        self._tfidf_cache_path = _CACHE_DIR / "tfidf.pkl"

        self._load_cache()

    # ── Кеширование ──────────────────────────────────────────

    def _load_cache(self) -> None:
        """Загрузить кеш индекса с диска."""
        if self._cache_path.exists() and self._tfidf_cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._chunks = data.get("chunks", [])
                self._indexed_files = data.get("indexed_files", {})

                with open(self._tfidf_cache_path, "rb") as f:
                    cached = pickle.load(f)
                self._vectorizer = cached["vectorizer"]
                self._tfidf_matrix = cached["matrix"]

                log.info(
                    "RAG кеш загружен: %d чанков, %d файлов",
                    len(self._chunks), len(self._indexed_files),
                )
            except Exception as exc:
                log.warning("Ошибка загрузки RAG кеша: %s", exc)
                self._chunks = []
                self._indexed_files = {}
                self._tfidf_matrix = None
                self._vectorizer = None

    def _save_cache(self) -> None:
        """Сохранить кеш индекса на диск."""
        try:
            data = {
                "chunks": self._chunks,
                "indexed_files": self._indexed_files,
            }
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            if self._vectorizer is not None and self._tfidf_matrix is not None:
                with open(self._tfidf_cache_path, "wb") as f:
                    pickle.dump({
                        "vectorizer": self._vectorizer,
                        "matrix": self._tfidf_matrix,
                    }, f)

            log.info("RAG кеш сохранён")
        except Exception as exc:
            log.error("Ошибка сохранения RAG кеша: %s", exc)

    # ── Индексация ───────────────────────────────────────────

    def _build_tfidf(self, texts: list[str]) -> None:
        """Построить TF-IDF матрицу из текстов."""
        if not texts:
            return

        log.info("Построение TF-IDF матрицы для %d текстов...", len(texts))
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        log.info("TF-IDF матрица: %s", self._tfidf_matrix.shape)

    def index_files(self, force: bool = False) -> int:
        """
        Проиндексировать все поддерживаемые файлы из data/.

        Parameters
        ----------
        force : bool
            Если True — переиндексировать всё, игнорируя кеш.

        Returns
        -------
        int
            Количество проиндексированных файлов.
        """
        if not DATA_DIR.exists():
            log.warning("Директория data/ не найдена")
            return 0

        files = [
            f for f in DATA_DIR.iterdir()
            if f.is_file()
            and f.suffix.lower() in _SUPPORTED_EXT
            and not f.name.startswith(".")
        ]

        if not files:
            log.info("Нет файлов для индексации")
            return 0

        new_chunks: list[dict] = []
        files_to_index: list[Path] = []
        unchanged_chunks: list[dict] = []

        for fp in files:
            current_hash = _file_hash(fp)
            cached_hash = self._indexed_files.get(fp.name)

            if not force and cached_hash == current_hash:
                unchanged_chunks.extend(
                    c for c in self._chunks if c["file"] == fp.name
                )
            else:
                files_to_index.append(fp)

        if not files_to_index and not force:
            log.info("Все файлы актуальны, индексация не требуется")
            return 0

        for fp in files_to_index:
            try:
                text = _extract_text(fp)
                if not text.strip():
                    log.warning("Пустой текст в %s — пропускаем", fp.name)
                    continue
                chunks = _split_chunks(text)
                for chunk in chunks:
                    new_chunks.append({"file": fp.name, "text": chunk})
                self._indexed_files[fp.name] = _file_hash(fp)
                log.info("Файл проиндексирован: %s (%d чанков)", fp.name, len(chunks))
            except Exception as exc:
                log.error("Ошибка индексации %s: %s", fp.name, exc)

        current_filenames = {f.name for f in files}
        self._indexed_files = {
            k: v for k, v in self._indexed_files.items() if k in current_filenames
        }

        all_chunks = unchanged_chunks + new_chunks

        if not all_chunks:
            log.info("Нет чанков после индексации")
            return 0

        texts = [c["text"] for c in all_chunks]
        self._build_tfidf(texts)
        self._chunks = all_chunks

        self._save_cache()

        log.info(
            "Индексация завершена: %d файлов, %d чанков",
            len(files_to_index), len(all_chunks),
        )
        return len(files_to_index)

    # ── Поиск ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Найти наиболее релевантные чанки для запроса.

        Parameters
        ----------
        query : str
            Поисковый запрос.
        top_k : int
            Количество результатов.

        Returns
        -------
        list[dict]
            Список {"file": str, "text": str, "score": float}.
        """
        if self._tfidf_matrix is None or self._vectorizer is None or len(self._chunks) == 0:
            log.warning("RAG индекс пуст. Сначала выполните index_files().")
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        for idx in indices:
            if scores[idx] > 0:
                results.append({
                    "file": self._chunks[idx]["file"],
                    "text": self._chunks[idx]["text"],
                    "score": float(scores[idx]),
                })

        log.info(
            "RAG поиск: '%s' → %d результатов (лучший: %.3f)",
            query[:50], len(results),
            results[0]["score"] if results else 0.0,
        )
        return results

    def build_context(self, query: str, top_k: int = TOP_K) -> str:
        """Построить контекстную строку для LLM из найденных чанков."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        parts: list[str] = [
            "Ниже — фрагменты из проиндексированных документов, "
            "релевантные запросу пользователя. "
            "Используй их для формирования точного ответа.\n"
        ]
        for i, r in enumerate(results, 1):
            parts.append(
                f"--- Фрагмент {i} (файл: {r['file']}, "
                f"релевантность: {r['score']:.2f}) ---\n{r['text']}\n"
            )

        return "\n".join(parts)

    @property
    def is_indexed(self) -> bool:
        """Есть ли данные в индексе."""
        return self._tfidf_matrix is not None and len(self._chunks) > 0

    @property
    def stats(self) -> dict:
        """Статистика индекса."""
        return {
            "files": len(self._indexed_files),
            "chunks": len(self._chunks),
            "indexed": self.is_indexed,
        }
