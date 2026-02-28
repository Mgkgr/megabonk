# Megabonk Bot

Проект для автоматизации игры Megabonk в двух режимах:
- `runtime` бот (правила + CV), основной сценарий запуска
- RL-окружение для обучения/инференса через Stable-Baselines3 (`train.py` / `play.py`)

## Что есть в репозитории

- `run_runtime_bot.py` — основной runtime-бот (hotkeys, overlay, лог событий)
- `megabonk_bot/config.py` — каноничные дефолты и валидация runtime-конфига
- `config/bot_profile.yaml` — пользовательский override-профиль runtime-бота
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

## Runtime-зависимости и производительность (важно)

В runtime и RL используются:
- `pydirectinput` для клавиатуры/мыши (DirectInput-совместимый ввод для DirectX-игр): [PyPI pydirectinput](https://pypi.org/project/PyDirectInput/)
- `WindowCapture` с backend `PrintWindow` (по умолчанию, захват привязан к окну) и fallback на `mss`: [python-mss docs](https://python-mss.readthedocs.io/)

Почему это важно:
- задержка реакции почти всегда складывается из `capture (PrintWindow/mss) + CV/логика + input (pydirectinput)`;
- `mss` чувствителен к DPI/масштабированию Windows, поэтому при `runtime.capture_backend: mss` несоответствие DPI даёт неверные ROI, «дрожащие» bbox и ложные лаги;
- `PrintWindow` позволяет читать кадр из окна даже если поверх экрана есть другие окна/оверлеи;
- у `pydirectinput` ввод работает ближе к игровому DirectInput-пути, но это не отменяет ограничений по частоте цикла и системной нагрузке.

Практические «числа производительности» для этого проекта:
- дефолт `runtime.step_hz = 12` -> целевой шаг `dt_ms ~= 83.3`;
- `runtime.event_log_interval_s = 0.2` -> запись агрегированных runtime-событий каждые 200 мс;
- если в `logs/runtime_events.jsonl` стабильно `latency_ms > dt_ms`, цикл перегружен (бот не успевает за своей частотой);
- рабочий ориентир для стабильного runtime: `p95(latency_ms) < dt_ms`, лучше с запасом 20-30%.

Мини-чек при симптомах лагов:
1. Проверить масштаб Windows (`Display Scale`) и запускать с одинаковым DPI, на котором снимались шаблоны.
2. Снизить `runtime.step_hz` (например, `12 -> 10 -> 8`) и сравнить `p95 latency_ms`.
3. Переснять `screen.png` и шаблоны на текущем разрешении/масштабе.

## Что подготовить перед первым запуском

1. Запустить игру и убедиться, что окно определяется по `window_title`.
2. Подготовить шаблоны в папке `templates/`.

### Подготовка шаблонов

1. Сделайте скриншот **всего окна игры** и сохраните как `screen.png` в корень проекта.
2. Запустите нарезку шаблонов:

```bash
python make_templates.py
```

Для автоматической нарезки HUD-полей под OCR (время, убийства, уровень, золото, hp):

```bash
python make_templates.py --export-hud --source screen.png --hud-outdir dbg_hud/screen_crops
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
- `--capture-backend auto|printwindow|mss` — принудительный backend захвата окна
- `--window-focus-interval-s <seconds>` — период принудительного возврата фокуса окну игры
- `--templates-dir <path>` — другая папка шаблонов
- `--print-default-config` — вывести каноничный YAML дефолтов и выйти

Логи runtime пишутся в `logs/runtime_events.jsonl`.

## Логи: структура JSONL события

События в `logs/runtime_events.jsonl` пишутся в формате JSONL (1 JSON-объект на строку).

- `schema_version`: версия схемы события (`runtime_events_v1` для текущей схемы)
- `ts`: UNIX timestamp (секунды)
- `mode`, `frame_id`, `screen`: состояние runtime-цикла
- `step_hz`, `dt_ms`: частота и шаг цикла (в миллисекундах)
- `window_title`: заголовок окна, из которого идёт захват
- `frame_size`: размер кадра (`width`, `height`)
- `telemetry`: HUD-метрики (`time`, `gold`, `lvl`, `kills`, `hp_ratio`)
- `telemetry_raw`: сырые OCR-метаданные HUD (`*_fail_reason`, `*_ocr_ms`, `*_rect`, `tesseract_cmd`)
- `latency_ms`: задержка итерации цикла

Укороченный пример строки:

```json
{"schema_version":"runtime_events_v1","ts":1730000000.12,"mode":"ACTIVE","frame_id":128,"step_hz":12,"dt_ms":83.33,"window_title":"Megabonk","frame_size":{"width":1280,"height":720},"telemetry":{"time":95.3,"gold":420,"lvl":4,"kills":12,"hp_ratio":0.76},"latency_ms":21.5}
```

Для экспорта JSONL в CSV используйте [`scripts/export_jsonl.py`](scripts/export_jsonl.py):

```bash
python scripts/export_jsonl.py logs/runtime_events.jsonl
python scripts/export_jsonl.py logs/runtime_events.jsonl --telemetry-only
```

Совместимость со старыми данными:
- если в строке нет `schema_version`, экспортёр трактует её как `legacy`.

## Конфигурация: единицы измерения и приоритеты

- Источник правды для дефолтов: `megabonk_bot/config.py` (`DEFAULT_CONFIG`).
- `config/bot_profile.yaml` и JSON-конфиг — только override поверх дефолтов.
- Приоритет применения: `DEFAULT_CONFIG` -> YAML/JSON override -> CLI-флаги (`--window`, `--templates-dir`, `--no-overlay`, `--no-hotkeys`).
- Для воспроизводимости можно получить эталонный YAML через `python run_runtime_bot.py --print-default-config`.

Единицы измерения:
- `_s` в названии поля (`event_log_interval_s`, `restart_*_s`, `scene_memory_ttl_s`) означает секунды.
- `_ticks` (`map_scan_interval_ticks`) означает тики основного цикла.
- `_pixels` (`cam_yaw_pixels`) означает пиксели мышиного смещения.
- `step_hz` — частота цикла в герцах (итераций в секунду).
- Пороги (`interact_threshold`) задаются в диапазоне `[0.0, 1.0]`.

### Тюнинг `autopilot` (YAML)

Пример секции:

```yaml
autopilot:
  click_cooldown_s: 0.5
  template_thresholds:
    main_play_detect: 0.60
    chest_pick: 0.65
  heuristic:
    stuck_frames_required: 6
    jump_cooldown: 30
    scan_interval: 60
```

Что значит:
- `click_cooldown_s` — минимальная пауза между кликами (`seconds`).
- `template_thresholds.*` — пороги template matching (`0.0..1.0`) для экранов/кнопок в `AutoPilot`.
- `heuristic.*` — параметры `HeuristicAutoPilot`:
  - `*_cooldown`, `stuck_escape_ticks`, `scan_interval`, `scan_duration`, `scan_decision_ticks`, `stuck_frames_required` задаются в `ticks` (итерациях цикла).
  - `enemy_hsv_*`, `coin_hsv_*` — HSV-диапазоны (`0..255`).
  - `enemy_area_threshold`, `coin_area_threshold` — площадь маски в пикселях.
  - `center_roi`, `center_lower_roi` — ROI в относительных долях кадра `[x, y, w, h]` (`0.0..1.0`).
  - `stuck_diff_threshold` — порог среднего diff по grayscale-кадрам.
  - `enemy_close_multiplier` — множитель “очень близкой” угрозы для решения о слайде.

### Как калибровать ROI времени HUD

- Базовый фиксированный ROI для времени: `HUD_TIME_RECT` в `megabonk_bot/hud.py`.
- Глобальный регион времени: `REG_HUD_TIME` в `megabonk_bot/regions.py`.
- По умолчанию текущие координаты фиксированного бокса: `(28, 61, 127, 41)`.
- Приоритет выбора ROI времени в рантайме:
  1. `regions["REG_HUD_TIME"]`, если передан в вызовы HUD.
  2. `HUD_TIME_RECT` из `megabonk_bot/hud.py`.
  3. Относительный ROI `DEFAULT_HUD_REGIONS["time"]` из `megabonk_bot/hud.py` (fallback для маленьких/нестандартных кадров).
- Практика калибровки: сначала обновите `REG_HUD_TIME` в `megabonk_bot/regions.py`, затем при необходимости скорректируйте `HUD_TIME_RECT` в `megabonk_bot/hud.py` и проверьте OCR через `read_hud_telemetry`.

## Что на что влияет (config/bot_profile.yaml)

### `runtime`
- `state`: стартовое состояние (`OFF`, `ACTIVE`, `PANIC`)
- `step_hz`: частота цикла бота; выше = быстрее реакция и выше нагрузка CPU
- `window_title`: какое окно захватывать
- `capture_backend`: backend захвата окна (`auto`, `printwindow`, `mss`)
- `window_focus_interval_s`: как часто принудительно проверять/возвращать фокус окна Megabonk
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

### `autopilot`
- `click_cooldown_s`: кулдаун клика для шаблонных действий `AutoPilot` (секунды)
- `template_thresholds`: точечный тюнинг порогов распознавания экранов/кнопок
- `heuristic`: параметры эвристического пилота (`UNSTUCK`, scan, jump/slide cooldown и пр.)

### `item_priorities`, `boss_schedule`
- Приоритеты предметов и расписание подготовки к боссам для MAX-логики.

## RL режим (опционально)

Обучение:

```bash
python train.py --window-title Megabonk --device cuda --timesteps 2000000
```

Ключевые PPO-гиперпараметры:
- `--n-steps` — длина rollout до обновления политики
- `--batch-size` — размер minibatch при SGD-обновлениях
- `--gamma` — discount factor для будущих наград
- описание параметров см. в официальной документации SB3 PPO: [stable-baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

Пространства окружения (`MegabonkEnv`) в терминах Gymnasium:
- `action_space`: [`gymnasium.spaces.MultiDiscrete`](https://gymnasium.farama.org/api/spaces/) (варианты зависят от `include_cam_yaw/include_cam_pitch`, см. docstring `MegabonkEnv`)
- `observation_space`: [`gymnasium.spaces.Box`](https://gymnasium.farama.org/api/spaces/) с `shape=(84, 84, frame_stack)`, `dtype=uint8`

Примечания:
- `--device` можно выбрать `cpu`, `cuda`, `cuda:0` и т.д.
- `--step-hz` фиксирует частоту шага окружения в `train.py`/`play.py` для воспроизводимых прогонов
- модель по умолчанию сохраняется в `megabonk_ppo_cnn.zip` (через `--model-out` можно изменить)

Инференс обученной модели:

```bash
python play.py --model-path megabonk_ppo_cnn --device cpu --window-title Megabonk
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

### Настройка Tesseract OCR (`pytesseract`)

Если `tesseract` не находится в `PATH`, задайте путь явно через переменную окружения `TESSERACT_CMD`.
В коде этот путь автоматически прокидывается в `pytesseract.pytesseract.tesseract_cmd`.

Пример (PowerShell):

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml
```

Для Windows обычно используют установщик Tesseract от UB Mannheim; после установки
можно либо добавить путь в `PATH`, либо использовать `TESSERACT_CMD`.

- Нестабильный детект/клики:
  - переснимите `screen.png` и шаблоны на текущем разрешении/DPI
  - уменьшите `step_hz`, если видите лаги

- `train.py` падает на CUDA:
  - установите совместимую сборку `torch` с CUDA
  - либо запустите на CPU через `python train.py --device cpu`
