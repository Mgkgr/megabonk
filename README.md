# Megabonk Bot

Простой практический гайд: как **настроить**, **запустить runtime-бота**, **обучить модель** и **запустить обученную модель**.

## 1) Что есть в проекте

- `run_runtime_bot.py` — основной runtime-бот (правила + CV + recovery).
- `train.py` — обучение PPO на `MegabonkEnv`.
- `play.py` — запуск уже обученной PPO-модели.
- `config/bot_profile.yaml` — ваш рабочий конфиг.
- `make_templates.py` — нарезка шаблонов и HUD-ROI из `screen.png`.

## 2) Быстрый сценарий (если хотите сразу запустить)

1. Установите зависимости.
2. Сделайте `screen.png` (полный скрин окна игры).
3. Нарежьте шаблоны (`python make_templates.py`).
4. Запустите runtime (`python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml`).
5. Включите бота: `F8` или кнопка `START` в оверлее.

Если нужен именно RL-режим, переходите в разделы `6` и `7`.

## 3) Требования

- ОС: **Windows** (управление и часть захвата завязаны на WinAPI/DirectInput).
- Python: **3.10+**.
- Игра Megabonk запущена в оконном/безрамочном режиме.
- Заголовок окна игры совпадает с `runtime.window_title` (по умолчанию `Megabonk`).

## 4) Установка

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# или cmd
.venv\Scripts\activate.bat

pip install -U pip
pip install numpy opencv-python mss pydirectinput gymnasium stable-baselines3 torch pillow pyyaml
```

Опционально:

- OCR HUD: `pip install pytesseract` + установленный Tesseract OCR.
- ONNX-детекция: `pip install onnxruntime`.

### Проверка установки

```bash
bash scripts/smoke_check.sh
pytest -q
```

## 5) Подготовка шаблонов и HUD

### 5.1 Нарезка шаблонов (обязательно для runtime)

1. Сохраните в корень проекта скрин окна игры: `screen.png`.
2. Запустите:

```bash
python make_templates.py
```

3. Выделите ROI мышью и задайте имя шаблона (например `tpl_play`).

Минимальный набор, с которым стоит начинать:

- `tpl_play`
- `tpl_confirm`
- `tpl_dead`
- `tpl_char_select_title`
- `tpl_lvl`
- `tpl_minimap`

### 5.2 Автоматическая нарезка HUD-полей для OCR (рекомендуется)

```bash
python make_templates.py --export-hud --source screen.png --hud-outdir dbg_hud/screen_crops
```

Что получите:

- `dbg_hud/screen_crops/time.png`
- `dbg_hud/screen_crops/kills.png`
- `dbg_hud/screen_crops/lvl.png`
- `dbg_hud/screen_crops/gold.png`
- `dbg_hud/screen_crops/hp.png`
- `dbg_hud/screen_crops/hud_preview.png`
- `dbg_hud/screen_crops/hud_regions.json`

## 6) Настройка перед запуском

Основной конфиг: `config/bot_profile.yaml`.

Минимум, который обычно правят:

- `runtime.window_title` — заголовок окна игры.
- `runtime.step_hz` — частота цикла (стартово 10-12).
- `runtime.capture_backend` — `auto` / `printwindow` / `mss`.
- `runtime.capture_log_errors` — логировать ли повторяющиеся ошибки захвата окна.
- `runtime.templates_dir` — папка шаблонов (обычно `templates`).
- `runtime.overlay_enabled` — показывать ли окно оверлея.
- `runtime.overlay_transparent` — прозрачный HUD-оверлей поверх окна игры.
- `runtime.hud_debug_save_policy` — политика сохранения HUD debug (`startup|on_fail_change|interval|off`).
- `runtime.hud_debug_min_interval_s` — минимальный интервал между повторными HUD debug-дампами.
- `runtime.event_schema_version` — версия схемы JSONL (`runtime_events_v2` по умолчанию, доступен `runtime_events_v1`).
- `hotkeys.toggle_vk` / `hotkeys.panic_vk` — hotkeys (по умолчанию F8/F12).

Если не уверены в значениях, оставьте дефолт и правьте только `window_title`.

## 7) Запуск runtime-бота (основной боевой режим)

```bash
python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml
```

Управление:

- `F8` — переключение `OFF <-> ACTIVE`.
- `F12` — `PANIC` (сброс зажатых клавиш).
- В оверлее: кнопки `START/STOP` и `PANIC` мышью.
- В оверлее: `S`/`Space` = toggle, `P` = panic, `Q`/`Esc` = выход.

Полезные флаги:

```bash
python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml --capture-backend printwindow
python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml --no-overlay
python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml --window-focus-interval-s 0.25
```

Логи:

- `logs/runtime_events.jsonl` — события runtime.
- `dbg_hud/` — debug-скрины HUD и кропы ROI (в том числе `time/kills/...`).

### Диагностика захвата

- В схеме `runtime_events_v2` в каждом событии есть блок `capture`:
  - `capture.bad_grab_count`
  - `capture.last_error`
- При повторяющихся проблемах захвата runtime пишет rate-limited warning в stdout.

### Политика HUD debug

- `runtime.hud_debug_save_policy: startup` — только стартовый дамп.
- `runtime.hud_debug_save_policy: on_fail_change` — дамп на первом фейле и при смене причины (рекомендуется).
- `runtime.hud_debug_save_policy: interval` — дамп при фейле, но не чаще `runtime.hud_debug_min_interval_s`.
- `runtime.hud_debug_save_policy: off` — отключить автоматические HUD debug-дампы.

В `runtime_events_v2` добавлен блок `hud`:
- `hud.debug_dumped` — был ли создан debug-дамп на текущем шаге.
- `hud.fail_streak` — текущая серия OCR-фейлов HUD.

## 8) Обучение модели (RL)

Базовый запуск:

```bash
python train.py --window-title Megabonk --device cuda --timesteps 2000000
```

Если CUDA недоступна:

```bash
python train.py --window-title Megabonk --device cpu --timesteps 500000
```

Часто используемые параметры:

- `--timesteps` — общее число шагов обучения.
- `--step-hz` — частота шага окружения.
- `--n-steps`, `--batch-size`, `--gamma`, `--lr` — гиперпараметры PPO.
- `--model-out` — имя/путь итоговой модели (без `.zip` или с путём).

Пример “быстрой проверки пайплайна”:

```bash
python train.py --window-title Megabonk --device cpu --timesteps 100000 --step-hz 8 --log-every 100
```

Результат обучения:

- файл модели, например `megabonk_ppo_cnn.zip`.
- tensorboard-логи в `tb/`.

## 9) Запуск обученной модели

```bash
python play.py --model-path megabonk_ppo_cnn --device cpu --window-title Megabonk
```

Полезно:

- `--stochastic` — стохастический выбор действий.
- `--step-hz` — синхронизация частоты с тем, на чём обучали.

## 10) Рекомендуемый рабочий цикл

1. Подготовить `screen.png` и шаблоны.
2. Проверить runtime-бот в реальной игре (детекты, recovery, HUD OCR).
3. Подкрутить `config/bot_profile.yaml` (частота, пороги, backend захвата).
4. Запустить обучение `train.py`.
5. Прогнать модель через `play.py`.
6. Повторить цикл с обновлёнными шаблонами/параметрами.

## 11) Частые проблемы и решения

### Не находится окно игры

- Проверьте `--window` и `runtime.window_title`.
- Убедитесь, что окно игры не свернуто.

### OCR HUD не работает

- Установите `pytesseract`.
- Установите Tesseract OCR.
- При необходимости задайте переменную `TESSERACT_CMD`.

PowerShell пример:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python run_runtime_bot.py --window Megabonk --config config/bot_profile.yaml
```

### Бот реагирует медленно

- Уменьшите `runtime.step_hz` (например 12 -> 10 -> 8).
- Переснимите `screen.png` и шаблоны на текущем разрешении/DPI.
- Попробуйте `--capture-backend printwindow`.

### Модель учится нестабильно

- Сначала уменьшите `step_hz` и проверьте стабильность эпизодов.
- Увеличьте `timesteps`.
- Сравните поведение на `--stochastic` и детерминированном запуске.

## 12) Полезные команды

Показать дефолтный конфиг:

```bash
python run_runtime_bot.py --print-default-config
```

Экспорт runtime JSONL в CSV:

```bash
python scripts/export_jsonl.py logs/runtime_events.jsonl
python scripts/export_jsonl.py logs/runtime_events.jsonl --telemetry-only
```
