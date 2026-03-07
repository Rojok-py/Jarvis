"""
Search — веб-поиск через DuckDuckGo.

Использует библиотеку duckduckgo-search для получения
результатов поиска без API-ключа.
"""

from __future__ import annotations

import logging

from duckduckgo_search import DDGS

log = logging.getLogger(__name__)

# Максимальное количество результатов по умолчанию
_MAX_RESULTS = 5


def web_search(query: str, max_results: int = _MAX_RESULTS) -> str:
    """
    Поиск в интернете через DuckDuckGo.

    Parameters
    ----------
    query : str
        Поисковый запрос.
    max_results : int
        Максимальное количество результатов.

    Returns
    -------
    str
        Форматированные результаты поиска.
    """
    if not query.strip():
        return "Пустой поисковый запрос."

    log.info("Веб-поиск: '%s' (макс. %d)", query, max_results)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"По запросу «{query}» ничего не найдено."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Без заголовка")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"{i}. **{title}**\n   {body}\n   🔗 {href}")

        output = "\n\n".join(lines)
        log.info("Найдено %d результатов", len(results))
        return output

    except Exception as exc:
        log.error("Ошибка веб-поиска: %s", exc)
        return f"Ошибка при поиске: {exc}"
