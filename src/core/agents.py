from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ключевые слова, требующие актуальных данных (триггер поиска)
_REALTIME_KEYWORDS = re.compile(
    r"(?:новост|погод|сегодня|вчера|сейчас|актуальн|последн|текущ|"
    r"weather|news|today|yesterday|current|latest|recent)",
    re.IGNORECASE,
)


def _get_llm(temperature: float = 0.4) -> ChatOpenAI:
    """Создать LLM, подключённый к Groq через OpenAI-совместимый API."""
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY не задан в .env файле.")

    return ChatOpenAI(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        openai_api_key=api_key,  # type: ignore[call-arg]
        openai_api_base="https://api.groq.com/openai/v1",  # type: ignore[call-arg]
        max_tokens=3000,
    )


def _search_web(query: str, max_results: int = 6) -> str:
    """DuckDuckGo поиск через langchain-community или прямой ddg SDK."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        tool = DuckDuckGoSearchRun()
        result = tool.run(query)
        log.info("DuckDuckGo (LangChain): '%s'", query[:80])
        return result
    except Exception:
        pass

    # Фолбэк на прямой duckduckgo-search SDK
    try:
        from duckduckgo_search import DDGS

        # Авто-определение региона: кириллица → ru-ru
        import re
        region = "ru-ru" if re.search(r"[\u0400-\u04ff]", query) else "wt-wt"

        with DDGS() as ddgs:
            results = list(ddgs.text(query, region=region, max_results=max_results))
        lines = []
        for i, r in enumerate(results, 1):
            body = r.get("body", "")
            href = r.get("href", "")
            title = r.get("title", "")
            lines.append(f"{i}. {title}\n   {body}\n   URL: {href}")
        return "\n\n".join(lines) if lines else "Результатов не найдено."
    except Exception as exc:
        log.error("Ошибка поиска: %s", exc)
        return f"Не удалось выполнить поиск: {exc}"


# ── Промпты ────────────────────────────────────────────────────────────────

_RESEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы — Research Agent J.A.R.V.I.S. Ваша задача: собрать полную"
            " и структурированную информацию по запросу пользователя."
            " Если предоставлены результаты поиска — используйте их."
            " Напишите подробный черновик ответа с фактами, деталями и источниками."
            " Отвечайте на том же языке, что и вопрос.",
        ),
        (
            "human",
            "Запрос: {query}\n\n"
            "{search_block}"
            "Напишите подробный черновик ответа:",
        ),
    ]
)

_EDITOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы — Editor Agent J.A.R.V.I.S., финальный редактор."
            " Вы получаете черновик от Research Agent и должны:"
            " 1) Убрать логические ошибки и противоречия."
            " 2) Сократить до сути, оставив ключевые факты."
            " 3) Отформатировать в фирменном стиле J.A.R.V.I.S.:"
            " вежливо, технологично, с лёгкой британской элегантностью."
            " 4) Обращаться к пользователю на 'сэр' (если он использовал русский)."
            " Отвечайте на том же языке, что и оригинальный вопрос.",
        ),
        (
            "human",
            "Оригинальный запрос: {query}\n\n"
            "Черновик Research Agent:\n{draft}\n\n"
            "Отредактируйте и дайте финальный ответ в стиле J.A.R.V.I.S.:",
        ),
    ]
)


# ── Основная функция ───────────────────────────────────────────────────────


def run_dual_agent(query: str) -> dict[str, str]:
    """
    Запустить двух-агентный пайплайн: Research → Editor.

    Parameters
    ----------
    query : str
        Вопрос или задача пользователя.

    Returns
    -------
    dict
        Словарь с ключами:
        - ``"research"``  — черновик Research Agent,
        - ``"final"``     — финальный ответ Editor Agent,
        - ``"searched"``  — True если выполнялся веб-поиск,
        - ``"search_results"`` — сырые результаты поиска (или "").
    """
    log.info("Dual-agent pipeline запущен: '%s'", query[:80])

    llm = _get_llm()
    parser = StrOutputParser()

    # ── Шаг 1: решить, нужен ли веб-поиск ─────────────────────────────
    needs_search = bool(_REALTIME_KEYWORDS.search(query))
    search_results = ""
    search_block = ""

    if needs_search:
        log.info("Запрос требует актуальных данных → веб-поиск")
        search_results = _search_web(query)
        search_block = (
            f"Результаты поиска DuckDuckGo:\n{search_results}\n\n"
        )

    # ── Шаг 2: Research Agent ──────────────────────────────────────────
    research_chain = _RESEARCH_PROMPT | llm | parser
    draft = research_chain.invoke(
        {"query": query, "search_block": search_block}
    )
    log.info("Research Agent завершён (%d символов)", len(draft))

    # ── Шаг 3: Editor Agent ───────────────────────────────────────────
    editor_chain = _EDITOR_PROMPT | llm | parser
    final = editor_chain.invoke({"query": query, "draft": draft})
    log.info("Editor Agent завершён (%d символов)", len(final))

    return {
        "research": draft,
        "final": final,
        "searched": needs_search,
        "search_results": search_results,
    }


def run_dual_agent_with_search(query: str) -> dict[str, str]:
    """
    Принудительный запуск с веб-поиском (для «найди с агентами»).
    """
    log.info("Dual-agent pipeline (forced search): '%s'", query[:80])

    llm = _get_llm()
    parser = StrOutputParser()

    search_results = _search_web(query)
    search_block = f"Результаты поиска DuckDuckGo:\n{search_results}\n\n"

    research_chain = _RESEARCH_PROMPT | llm | parser
    draft = research_chain.invoke(
        {"query": query, "search_block": search_block}
    )

    editor_chain = _EDITOR_PROMPT | llm | parser
    final = editor_chain.invoke({"query": query, "draft": draft})

    return {
        "research": draft,
        "final": final,
        "searched": True,
        "search_results": search_results,
    }
