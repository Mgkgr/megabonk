# Megabonk Bot

Бот для Megabonk под Windows. Основной режим сейчас - runtime-бот: захват окна игры, CV/шаблоны, HUD/objective OCR, memory probe, навигация, recovery после смерти, overlay и JSONL-телеметрия. RL-часть на `stable-baselines3` осталась в проекте как экспериментальный режим через `MegabonkEnv`.

Проект управляет реальным окном игры через `pydirectinput`, поэтому перед запуском проверьте hotkeys и держите под рукой `F12` panic.

## Что внутри

- `run_runtime_bot.py` - основной runtime-бот.
- `config/survival_mvp.yaml` - рекомендуемый survival-only профиль runtime.
- `megabonk_bot/` - пакет с распознаванием, конфигом, runtime-логикой, overlay, navigation, memory probe и event logger.
- `templates/` - PNG-шаблоны UI/HUD.
- `art_refs/megabonk_unity_extracts/` - референсы и иконки для каталогов.
- `config/*_catalog.json`, `config/ocr_lexicon.json` - каталоги врагов, объектов, снарядов и OCR-лексикон.
- `make_templates.py` - нарезка шаблонов и HUD ROI из скриншота.
- `scripts/profile_runtime_budget.py` - baseline производительности runtime-стадий.
- `scripts/export_jsonl.py` - экспорт `logs/runtime_events.jsonl` в CSV.
- `scripts/smoke_memory_probe.py` - быстрая проверка memory probe на живом окне игры.
- `train.py`, `play.py`, `megabonk_env.py` - RL-режим.
- `tests/` - unit-тесты.

## Требования

- Windows.
- Python 3.10+.
- Megabonk в оконном или borderless-режиме.
- Заголовок окна должен содержать `Megabonk` или значение из `runtime.window_title`.

Зависимости пока не зафиксированы в `requirements.txt`, поэтому установка ручная:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install numpy opencv-python mss pydirectinput gymnasium stable-baselines3 torch pillow pyyaml pytest
```

Опционально:

```powershell
pip install pytesseract onnxruntime pandas UnityPy
```

- `pytesseract` нужен для HUD/objective OCR, плюс должен быть установлен Tesseract OCR.
- `onnxruntime` нужен только при `detection.use_onnx: true`.
- `pandas` нужен только для `scripts/export_jsonl.py --use-pandas`.
- `UnityPy` нужен только для `scripts/extract_megabonk_refs.py`.

Если Tesseract установлен не в стандартном месте:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Быстрый запуск runtime

1. Запустите игру и оставьте окно видимым.
2. Обновите `screen.png`: положите в корень проекта свежий полный скрин окна игры.
3. При необходимости нарежьте шаблоны:

```powershell
python make_templates.py --source screen.png --outdir templates
```

4. Обновите HUD ROI для диагностики OCR:

```powershell
python make_templates.py --export-hud --source screen.png --hud-outdir dbg_hud/screen_crops
```

5. Запустите runtime:

```powershell
python run_runtime_bot.py --window Megabonk --config config/survival_mvp.yaml
```

Управление:

- `F8` - переключить `OFF` / `ACTIVE`.
- `F12` - `PANIC`, отпустить зажатые клавиши.
- В overlay: `START/STOP`, `PANIC`.
- В окне overlay: `S` или `Space` - toggle, `P` - panic, `Q` или `Esc` - выход.

Полезные варианты запуска:

```powershell
python run_runtime_bot.py --window Megabonk --config config/survival_mvp.yaml --capture-backend printwindow
python run_runtime_bot.py --window Megabonk --config config/survival_mvp.yaml --capture-backend mss
python run_runtime_bot.py --window Megabonk --config config/survival_mvp.yaml --no-overlay
python run_runtime_bot.py --window Megabonk --config config/survival_mvp.yaml --templates-dir templates
python run_runtime_bot.py --print-default-config
```

## Конфиг

Рекомендуемый файл для runtime: `config/survival_mvp.yaml`.

Чаще всего правятся:

- `runtime.window_title` - строка поиска окна.
- `runtime.step_hz` - частота runtime-цикла.
- `runtime.capture_backend` - `auto`, `printwindow` или `mss`.
- `runtime.templates_dir` - папка шаблонов.
- `runtime.overlay_enabled`, `runtime.overlay_transparent`, `runtime.overlay_topmost` - режим overlay.
- `runtime.hud_ocr_every_s` - частота фонового HUD OCR.
- `runtime.objective_ocr_every_s` - частота фонового objective OCR; есть в дефолтном конфиге, даже если не указан в `bot_profile.yaml`.
- `runtime.event_log_path` - JSONL-лог.
- `runtime.event_schema_version` - по умолчанию `runtime_events_v4`.
- `runtime.performance_budget_enabled` и `runtime.performance_budget_ms` - бюджет стадий `capture`, `hud`, `scene_analysis`, `overlay`.
- `hotkeys.toggle_vk`, `hotkeys.panic_vk` - virtual-key коды hotkeys.
- `detection.asset_refs_dir`, `detection.enemy_catalog_path`, `detection.world_catalog_path`, `detection.projectile_catalog_path`, `detection.ocr_lexicon_path` - ресурсы распознавания.
- `detection.memory_probe_enabled`, `detection.memory_signatures_path`, `detection.memory_poll_interval_s` - memory probe.
- `navigation.profile` - `cautious`, `balanced` или `aggressive`.

Относительные пути из конфига резолвятся от корня проекта, найденного по `--config`.

## Runtime-лог

Runtime пишет события в:

```text
logs/runtime_events.jsonl
```

Текущая схема по умолчанию - `runtime_events_v4`. В событии есть:

- `capture` - ошибки и счётчик плохих кадров.
- `hud` - debug dump и streak OCR-фейлов.
- `telemetry` и `telemetry_raw` - HUD-значения и диагностические OCR-поля.
- `detections` - враги, снаряды, объекты, hazards и классы из каталогов.
- `player_pose`, `map_state`, `memory_probe_status` - данные memory probe и карты, если доступны.
- `navigation` - оценка полос, terrain/drop risk/clearance и gates для jump/slide.
- `performance` - timings стадий и превышения бюджета.

Экспорт в CSV:

```powershell
python scripts/export_jsonl.py logs/runtime_events.jsonl
python scripts/export_jsonl.py logs/runtime_events.jsonl --telemetry-only
```

## Производительность

Для baseline по `screen.png`:

```powershell
python scripts/profile_runtime_budget.py --config config/bot_profile.yaml --sample-frame screen.png --iterations 20
```

Для живого окна:

```powershell
python scripts/profile_runtime_budget.py --config config/bot_profile.yaml --live-window Megabonk --capture-backend auto --iterations 20
```

Результат сохраняется в `logs/runtime_performance_baseline.json`. Runtime также пишет throttled warning, если `performance_budget_enabled: true` и стадия выходит за бюджет.

## Memory Probe

Memory probe читает состояние внешнего процесса через WinAPI. Если `config/memory_signatures.json` пустой или сигнатуры не подходят, runtime продолжит работу в degraded-режиме, а в событиях будет соответствующий `memory_probe_status`.

Проверка:

```powershell
python scripts/smoke_memory_probe.py --window Megabonk --signatures config/memory_signatures.json
```

Важная деталь: `memory_probe` и `WindowCapture` ищут одно и то же видимое окно: сначала точное совпадение без учёта регистра, затем самый короткий заголовок, содержащий заданную строку. Если probe не видит процесс, проверьте `runtime.window_title`/`--window` и `detection.memory_signatures_path`.

## Шаблоны и HUD

Минимальный набор шаблонов для runtime:

- `tpl_play`
- `tpl_confirm`
- `tpl_dead`
- `tpl_char_select_title`
- `tpl_lvl`
- `tpl_minimap`

В `templates/` уже лежат рабочие шаблоны, но после смены разрешения, DPI или UI-скейла лучше обновить `screen.png` и переснять проблемные ROI.

HUD debug export создаёт:

- `dbg_hud/screen_crops/time.png`
- `dbg_hud/screen_crops/kills.png`
- `dbg_hud/screen_crops/lvl.png`
- `dbg_hud/screen_crops/gold.png`
- `dbg_hud/screen_crops/hp.png`
- `dbg_hud/screen_crops/hud_preview.png`
- `dbg_hud/screen_crops/hud_regions.json`

## RL-режим

Обучение PPO:

```powershell
python train.py --window-title Megabonk --device cuda --timesteps 2000000
```

CPU-проверка пайплайна:

```powershell
python train.py --window-title Megabonk --device cpu --timesteps 100000 --step-hz 8 --log-every 100
```

Запуск обученной модели:

```powershell
python play.py --model-path megabonk_ppo_cnn --device cpu --window-title Megabonk
python play.py --model-path megabonk_ppo_cnn --device cpu --window-title Megabonk --stochastic
```

Модель сохраняется как `megabonk_ppo_cnn.zip`, TensorBoard-логи - в `tb/`.

## Проверки

Unit-тесты:

```powershell
pytest -q
```

Smoke-check синтаксиса:

```bash
bash scripts/smoke_check.sh
```

`scripts/smoke_check.sh` рассчитан на Bash и `rg`; на Windows удобнее запускать его через Git Bash или WSL.

## Частые проблемы

Окно не найдено:

- Проверьте `--window` и `runtime.window_title`.
- Окно игры должно быть видимым, не свёрнутым.
- Capture и memory probe используют одну стратегию поиска: точное совпадение без учёта регистра, затем shortest substring-match среди видимых окон.

Чёрный или пустой кадр:

- Попробуйте `--capture-backend printwindow`.
- Попробуйте `--capture-backend mss`.
- Перезапустите игру в borderless/windowed.
- Проверьте DPI и переснимите `screen.png`.

OCR не даёт значения:

- Установите Tesseract OCR и `pytesseract`.
- Проверьте `$env:TESSERACT_CMD`.
- Сгенерируйте HUD crops через `make_templates.py --export-hud`.
- Посмотрите `telemetry_raw` и `hud.fail_streak` в JSONL.

Runtime не держит `step_hz`:

- Запустите `scripts/profile_runtime_budget.py`.
- Уменьшите `runtime.step_hz`.
- Увеличьте `runtime.hud_ocr_every_s` и `runtime.objective_ocr_every_s`.
- Отключите overlay через `--no-overlay` для сравнения.

Неправильные детекты:

- Переснимите шаблоны под текущие разрешение и DPI.
- Проверьте каталоги в `config/*_catalog.json`.
- Для врагов настройте `detection.enemy_hsv_lower`, `detection.enemy_hsv_upper`, `detection.enemy_min_area`.

## Текущие ограничения

- Нет зафиксированного dependency manifest; установка пока ручная.
- Memory signatures требуют актуальных адресов/паттернов под текущую версию игры.
- ONNX-детекция выключена по умолчанию.
- RL-режим зависит от стабильности реального окна игры и не является быстрым offline-тренажёром.
