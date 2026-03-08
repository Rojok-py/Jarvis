"""
Search — веб-поиск через DuckDuckGo + погода через wttr.in.

Использует библиотеку duckduckgo-search для получения
результатов поиска без API-ключа.
Для погоды — бесплатный сервис wttr.in (без ключа, без зависимостей).
"""

from __future__ import annotations

import logging
import re
import urllib.request
import urllib.error
import json

from duckduckgo_search import DDGS

log = logging.getLogger(__name__)

# Максимальное количество результатов по умолчанию
_MAX_RESULTS = 8

# Паттерн для определения кириллического текста
_HAS_CYRILLIC = re.compile(r"[\u0400-\u04ff]")

# Паттерн для определения запроса о погоде и извлечения города
_WEATHER_PATTERN = re.compile(
    r"(?:погод[аеуыёо]|температур[аеуы]|прогноз\s+погоды|weather|forecast)"
    r".*?(?:в\s+|in\s+|для\s+|for\s+)?(?P<city>[A-Za-z\u0400-\u04ff][A-Za-z\u0400-\u04ff\s\-]{1,30})?",
    re.IGNORECASE,
)


def _is_russian(text: str) -> bool:
    """Определить, содержит ли текст кириллицу."""
    return bool(_HAS_CYRILLIC.search(text))


def _extract_weather_city(query: str) -> str | None:
    """Извлечь город из запроса о погоде. Возвращает None если не запрос о погоде."""
    m = _WEATHER_PATTERN.search(query)
    if not m:
        return None
    city = (m.group("city") or "").strip().rstrip(".,;:!?")
    return city if city else "Москва"


def get_weather(city: str) -> str:
    """
    Получить текущую погоду через wttr.in.

    Parameters
    ----------
    city : str
        Название города.

    Returns
    -------
    str
        Форматированная строка с данными о погоде.
    """
    log.info("Запрос погоды: '%s' через wttr.in", city)
    try:
        # JSON-формат для структурированных данных
        url = f"https://wttr.in/{urllib.request.quote(city)}?format=j1&lang=ru"
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]

        area_name = city
        if area:
            names = area.get("areaName", [{}])
            if names:
                area_name = names[0].get("value", city)

        temp_c = current.get("temp_C", "?")
        feels_like = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        wind_speed = current.get("windspeedKmph", "?")
        wind_dir = current.get("winddir16Point", "")
        pressure = current.get("pressure", "?")
        visibility = current.get("visibility", "?")

        # Описание погоды на русском
        desc_list = current.get("lang_ru", [])
        if desc_list:
            description = desc_list[0].get("value", "")
        else:
            description = current.get("weatherDesc", [{}])[0].get("value", "")

        # Прогноз на сегодня
        forecast_today = data.get("weather", [{}])[0]
        max_temp = forecast_today.get("maxtempC", "?")
        min_temp = forecast_today.get("mintempC", "?")

        result = (
            f"Погода в {area_name}:\n"
            f"🌡 Температура: {temp_c}°C (ощущается как {feels_like}°C)\n"
            f"📝 {description}\n"
            f"🌡 Мин/Макс за день: {min_temp}°C / {max_temp}°C\n"
            f"💧 Влажность: {humidity}%\n"
            f"💨 Ветер: {wind_speed} км/ч {wind_dir}\n"
            f"📊 Давление: {pressure} мбар\n"
            f"👁 Видимость: {visibility} км"
        )

        log.info("Погода получена для %s: %s°C", area_name, temp_c)
        return result

    except urllib.error.HTTPError as exc:
        log.error("wttr.in HTTP ошибка: %s", exc)
        return f"Не удалось получить погоду для «{city}»: HTTP {exc.code}"
    except Exception as exc:
        log.error("Ошибка получения погоды: %s", exc)
        return f"Ошибка при получении погоды для «{city}»: {exc}"


def web_search(query: str, max_results: int = _MAX_RESULTS) -> str:
    """
    Поиск в интернете через DuckDuckGo.
    Для запросов о погоде автоматически использует wttr.in.

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

    # Проверяем: запрос о погоде → используем wttr.in
    weather_city = _extract_weather_city(query)
    if weather_city:
        return get_weather(weather_city)

    # Автоопределение региона по языку запроса
    region = "ru-ru" if _is_russian(query) else "wt-wt"

    log.info("Веб-поиск: '%s' (регион=%s, макс. %d)", query, region, max_results)

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, region=region, max_results=max_results)
            )

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
