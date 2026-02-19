# Megabonk Bot

Проект для автоматизации игры Megabonk в двух режимах:
- `runtime` бот (правила + CV), основной сценарий запуска
- RL-окружение для обучения/инференса через Stable-Baselines3 (`train.py` / `play.py`)

## Что есть в репозитории

- `run_runtime_bot.py` — основной runtime-бот (hotkeys, overlay, лог событий)
- `config/bot_profile.yaml` — профиль runtime-бота
- `make_templates.py` — интерактивная нарезка шаблонов из `screen.png`
- `train.py` — обучение PPO
- `play.py` — запуск обученной PPO-модели
- `scripts/smoke_check.sh` — быстрая проверка синтаксиса и конфликт-маркеров

## Требования

- ОС: **Windows** (управление и захват окна завязаны на WinAPI/pydirectinput)
- Python: **3.10+**
- Запущенная игра Megabonk в оконном/безрамочном режиме
- Заголовок окна должен совпадать с `window_title` в конфиге (по умолчанию `Megabonk`)

## Установка

1. Создайте и активируйте виртуальное окружение.

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# или cmd
.venv\Scripts\activate.bat
```

2. Установите зависимости.

```bash
pip install -U pip
pip install numpy opencv-python mss pydirectinput gymnasium stable-baselines3 torch pillow pyyaml
```

3. Опциональные зависимости:
- OCR HUD: `pip install pytesseract` + установленный Tesseract OCR
- ONNX-детектор (`detection.use_onnx: true`): `pip install onnxruntime`

## Что подготовить перед первым запуском

1. Запустить игру и убедиться, что окно определяется по `window_title`.
2. Подготовить шаблоны в папке `templates/`.

### Подготовка шаблонов

1. Сделайте скриншот **всего окна игры** и сохраните как `screen.png` в корень проекта.
2. Запустите нарезку шаблонов:

```bash
python make_templates.py
```

3. В окне выделяйте ROI мышью и задавайте имена шаблонов.
4. Минимально полезные шаблоны:
- `tpl_play`
- `tpl_confirm`
- `tpl_dead`
- `tpl_char_select_title`
- `tpl_lvl`
- `tpl_minimap`

Чем полнее набор шаблонов, тем стабильнее поведение в меню/рекавери.

## Быстрый старт (рекомендуемый)

Запуск runtime-бота:

```bash
python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml
```

Управление по умолчанию:
- `F8` — переключение `OFF/ACTIVE`
- `F12` — `PANIC` и отпускание зажатых клавиш

Полезные флаги:
- `--no-overlay` — без окна визуализации
- `--no-hotkeys` — без WinAPI hotkeys
- `--templates-dir <path>` — другая папка шаблонов

Логи runtime пишутся в `logs/runtime_events.jsonl`.

## Что на что влияет (config/bot_profile.yaml)

### `runtime`
- `state`: стартовое состояние (`OFF`, `ACTIVE`, `PANIC`)
- `step_hz`: частота цикла бота; выше = быстрее реакция и выше нагрузка CPU
- `window_title`: какое окно захватывать
- `templates_dir`: откуда грузить шаблоны
- `overlay_enabled`, `overlay_window`, `overlay_topmost`: поведение overlay
- `event_log_path`, `event_log_interval_s`: путь и частота записи JSONL-логов
- `cam_yaw_pixels`: амплитуда поворота камеры по yaw
- `restart_*`: логика восстановления после смерти (кулдаун, длительность удержания `R`, таймаут, число попыток)

### `hotkeys`
- `enabled`: включить/выключить опрос горячих клавиш
- `toggle_vk`, `panic_vk`: VK-коды клавиш (по умолчанию F8/F12)

### `detection`
- `grid_rows`, `grid_cols`: разбиение кадра для оценки препятствий
- `enemy_hsv_lower/upper`, `enemy_min_area`: чувствительность детектора врагов
- `interact_threshold`: порог template matching интерактивных объектов
- `use_onnx`, `onnx_model_path`: подключение ONNX-модели
- `scene_memory_ttl_s`: TTL памяти детекций

### `mvp_policy`
- `chest_policy`: стратегия работы с сундуками
- `auto_pick_upgrade_with_space`: автопик улучшений пробелом
- `user_picks_character_manually`: ручной выбор персонажа
- `allow_map_scan_tab`, `map_scan_interval_ticks`: скан карты через `Tab`

### `max_policy`
- `enabled`: включение расширенной логики MAX
- Остальные поля (`threat_scoring`, `bunny_hop_enabled`, и т.д.) включают отдельные подсистемы поведения

### `item_priorities`, `boss_schedule`
- Приоритеты предметов и расписание подготовки к боссам для MAX-логики.

## RL режим (опционально)

Обучение:

```bash
python train.py
```

Важно:
- в `train.py` принудительно используется `device="cuda"`; без CUDA будет ошибка
- модель сохраняется в `megabonk_ppo_cnn.zip`

Инференс обученной модели:

```bash
python play.py
```

`play.py` ожидает файл модели `megabonk_ppo_cnn.zip` в корне проекта.

## Проверки перед запуском

Smoke-check:

```bash
bash scripts/smoke_check.sh
```

Тесты:

```bash
pytest -q
```

## Типовые проблемы

- Не находится окно игры:
  - проверьте `runtime.window_title` и `--window`
  - убедитесь, что игра запущена и окно не свернуто

- Нет OCR-значений HUD:
  - установите `pytesseract`
  - установите Tesseract OCR и при необходимости задайте `TESSERACT_CMD`

- Нестабильный детект/клики:
  - переснимите `screen.png` и шаблоны на текущем разрешении/DPI
  - уменьшите `step_hz`, если видите лаги

- `train.py` падает на CUDA:
  - установите совместимую сборку `torch` с CUDA
  - либо адаптируйте `train.py` под CPU (сейчас в коде стоит жесткая проверка CUDA)
