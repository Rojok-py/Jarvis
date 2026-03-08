# J.A.R.V.I.S.

Голосовой и текстовый AI-ассистент для Arch Linux на базе Groq (Llama 3.3 70B).  
Поддерживает RAG по документам, веб-поиск, анализ файлов, запись голоса.

---

## Стек

| | |
|---|---|
| LLM | Llama 3.3 70B — Groq API |
| STT | Whisper Large V3 Turbo — Groq API |
| TTS | edge-tts + mpv |
| Embeddings | TF-IDF (scikit-learn, локально) |
| RAG | PyMuPDF + cosine similarity |
| UI | PyQt6 |
| Запись | FFmpeg + PulseAudio |
| Поиск | DuckDuckGo |
| Python | 3.12+, Poetry |

---

## Структура

```
src/
├── main.py              — точка входа
├── audio/
│   ├── recorder.py      — запись через FFmpeg (удержание кнопки)
│   └── speaker.py       — TTS через edge-tts + mpv
├── core/
│   ├── engine.py        — Groq: чат, аудио, RAG, файлы, команды
│   └── rag.py           — TF-IDF индекс, чанкинг, поиск
├── tools/
│   ├── file_ops.py      — работа с data/ и out/
│   └── search.py        — веб-поиск DuckDuckGo
└── ui/
    ├── interface.py     — главное окно, режимы, воркеры
    └── animations.py    — GIF-виджет
assets/                  — jarvis.gif, jarvis.webp
data/                    — файлы для RAG и анализа
out/                     — результаты анализа
.rag_cache/              — кеш TF-IDF индекса (авто)
```

---

## Установка

```bash
# системные зависимости
sudo pacman -S ffmpeg mpv python python-pip

# Poetry
curl -sSL https://install.python-poetry.org | python3 -

# проект
git clone https://github.com/Rojok-py/Jarvis.git && cd Jarvis
poetry install

# API-ключ
echo "GROQ_API_KEY=ваш_ключ" > .env
```

Ключ: [console.groq.com/keys](https://console.groq.com/keys)

---

## Запуск

```bash
poetry run python src/main.py
```

---

## Использование

### Режимы
- **ТЕКСТ** — вводить сообщение, Enter или кнопка «Отправить»
- **ГОЛОС** — переключить кнопкой «Режим», затем **удерживать 🎤** пока говоришь, отпустить — запись отправится на распознавание и озвучку

Кнопка **Стоп** прерывает ответ и TTS в любой момент в обоих режимах.

### Команды

| Команда | Действие |
|---|---|
| `проанализируй файл report.pdf` | анализ файла через Groq |
| `проанализируй файл report.pdf и сохрани в summary.txt` | то же + сохранить в out/ |
| `найди в интернете ...` / `поищи ...` | DuckDuckGo + Groq |
| `покажи файлы` / `список файлов` | листинг data/ и out/ |
| `по документам ...` / `по файлам ...` | RAG-запрос |
| `индексируй файлы` / `обнови индекс` | переиндексировать data/ |
| `создай файл notes.txt ...` | создать файл в data/ |
| `измени файл notes.txt ...` | отредактировать через LLM |
| `допиши в файл notes.txt ...` | дополнить файл |
| `сброс чата` / `новый чат` | очистить историю |

### RAG

1. Положи файлы в `data/` — PDF, TXT, MD, CSV, JSON, PY
2. Нажми **📑 Индексация** (или скажи «индексируй файлы»)
3. Спрашивай: `по документам что такое алгоритм?`

RAG разбивает файлы на чанки (800 символов / перекрытие 200), строит TF-IDF матрицу, при запросе ищет топ-5 по косинусному сходству и передаёт контекст в Groq.  
Индекс кешируется и пересчитывается только при изменении файлов.

---

## Модули

**`engine.py`** — `JarvisEngine`:
- `send_text()` — чат с историей
- `send_audio()` — Whisper → команды / чат
- `analyze_file()` — читает файл, отдаёт в Groq
- `rag_query()` — RAG + Groq
- `index_documents()` — индексация data/
- `process_voice_command()` — маршрутизация по интентам
- `_handle_create_file()` / `_handle_edit_file()` / `_handle_append_file()` — файловые операции через LLM
- `reset_chat()` — сброс истории

**`rag.py`** — `RAGIndex`:
- парсинг PDF (PyMuPDF) и текстовых файлов
- чанкинг с перекрытием
- TF-IDF через scikit-learn, косинусное сходство через NumPy
- инкрементальное кеширование (JSON + .npy)

**`recorder.py`**:
- `start_recording()` — запускает ffmpeg без ограничения, возвращает процесс
- `stop_recording()` — SIGTERM → ffmpeg корректно закрывает wav
- `record(duration)` — запись фиксированной длины

**`speaker.py`**:
- `speak()` — edge-tts генерирует mp3, mpv воспроизводит
- `stop()` — убивает mpv (используется кнопкой Стоп)
- очищает markdown и эмодзи перед озвучкой


