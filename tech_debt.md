# Tech Debt: Megabonk Bot

Дата анализа: 2026-05-20.

## Короткий диагноз

- Текущий runtime перегружен несколькими конкурирующими направлениями: HSV-детект врагов, каталоги ассетов, memory probe, minimap, OCR, navigation planner, overlay, recovery и старый RL-контур.
- Минимальная полезная цель сейчас: survival MVP. Бот должен видеть ближайших врагов/препятствия, бежать от давления и не делать лишних действий.
- Активный `config/bot_profile.yaml` использует JSON-каталоги, но значительная часть путей в `config/enemy_catalog.json`, `config/world_catalog.json` и `config/projectile_catalog.json` указывает на отсутствующие `art_refs/.../enemies`, `meshes`, `silhouettes`, `minimap_icons`.
- Свежие ручные скрины уже лежат в `art_refs/megabonk_unity_extracts/imported/...` и описаны в YAML-каталогах, но активный профиль их пока не использует.
- На `screen.png` текущий HSV-детект врагов дает большие ложные боксы, включая почти всю нижнюю часть экрана. Это напрямую может объяснять хаотичное движение.
- Проверка на момент анализа: `.\.venv\Scripts\python.exe -m pytest -q` -> `121 passed`.

## Подтвержденные факты из проекта

- Директория игры доступна: `G:\SteamLibrary\steamapps\common\Megabonk\Megabonk.exe` и `Megabonk_Data` найдены.
- Активный профиль сейчас грузит JSON-каталоги:
  - enemies: 25 записей, previews есть только у 1;
  - world: 28 записей, previews есть у 15;
  - projectiles: 15 записей, previews нет ни у одной.
- Ручные YAML-каталоги сейчас ближе к реальному состоянию:
  - enemies: 1 запись, preview есть;
  - world: 2 записи, previews есть у обеих;
  - projectiles: 15 записей, previews нет.
- `tb/PPO_1/events.out.tfevents.*` уже tracked в git и должен быть удален из репозитория как артефакт обучения.
- Корневые legacy-файлы `megabonk_env.py`, `train.py`, `play.py`, `regions.py`, `vision.py` тоже tracked; их нельзя удалять вслепую, но они должны пройти отдельную задачу на архив/удаление.

## Первый порядок работ

1. Перестать расширять текущий "все-и-сразу" runtime и зафиксировать survival MVP как единственный ближайший режим.
2. Перенести полезные ручные скрины из `dbg_hud` в `art_refs/megabonk_unity_extracts/imported`.
3. Сделать активный survival-каталог только из реально существующих PNG/JPG.
4. Запретить огромные HSV-боксы и добавить smoke-тест на `screen.png`.
5. Упростить policy до "беги от угрозы, иначе вперед"; прыжки, slide, сундуки, лут, minimap и memory probe не использовать в MVP.
6. После стабилизации survival MVP удалить или архивировать RL-контур, TensorBoard-артефакты и корневые дубли.

## P0 - Survival MVP

- [x] Зафиксировать один целевой режим `survival_mvp`.
  - Оставить в hot path только захват кадра, screen state, враги, простые препятствия, выбор направления, panic/recovery.
  - Не использовать для MVP сбор лута, сундуки, shrine, minimap, projectiles, boss prep, map scan и memory-dependent действия.
  - Критерий готовности: есть отдельный профиль или явный режим запуска, где бот только выживает и не переключается в побочные цели.
  - [x] Добавить отдельный профиль `config/survival_mvp.yaml` с survival-only флагами.
    - changed files: `config/survival_mvp.yaml`, `tests/test_config.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
    - limitations/follow-up: профиль создан и проверен, но активный `config/bot_profile.yaml` и runtime path ещё не переведены на него; родительскую задачу оставить открытой до переключения запуска и проверки поведения в игре.
  - [x] Перевести дефолтный runtime launch на `config/survival_mvp.yaml`.
    - changed files: `run_runtime_bot.py`, `tests/test_config.py`, `README.md`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
    - limitations/follow-up: дефолтный CLI и документация переведены на survival-only профиль, но родительская задача остаётся открытой до подтверждения поведения в живой игре и отделения runtime от legacy `config/bot_profile.yaml`.
  - [x] Отключить objective OCR и boss-prep ветку в `survival_only` runtime path.
    - changed files: `run_runtime_bot.py`, `megabonk_bot/config.py`, `config/survival_mvp.yaml`, `tests/test_config.py`, `tests/test_runtime_objective_cache.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: survival-only path теперь не поднимает objective OCR cache и не вычисляет boss prep, но в hot path всё ещё остаются legacy-зависимости `AutoPilot`/HUD, которые нужно сужать отдельными шагами.
  - [x] Отключить memory probe, bunny-hop/slide и side-goal флаги в fallback `config/bot_profile.yaml`.
    - changed files: `config/bot_profile.yaml`, `tests/test_config.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
    - limitations/follow-up: fallback-профиль теперь не включает memory-dependent и movement-side-goal флаги, но родительская задача остаётся открытой, пока hot path всё ещё зависит от legacy `AutoPilot`/HUD и не подтверждён в живой игре.
  - [x] Принудительно запретить `TAB` map scan и `SPACE` auto-pick в `survival_only` runtime path даже при конфликтных флагах конфига.
    - changed files: `run_runtime_bot.py`, `tests/test_runtime_objective_cache.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: guard теперь режет два побочных действия на уровне runtime, но родительская задача остаётся открытой, пока hot path всё ещё зависит от legacy `AutoPilot`/HUD и не подтверждён в живой игре.
  - [x] Отключить HUD OCR cache в `survival_only` runtime path.
    - changed files: `run_runtime_bot.py`, `tests/test_runtime_objective_cache.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: survival-only path больше не поднимает HUD OCR поток и читает пустую telemetry snapshot, но родительская задача остаётся открытой, пока runtime всё ещё зависит от legacy `AutoPilot` для screen state/recovery и не подтверждён в живой игре.
  - [x] Вынести screen-state/death helpers из `autopilot.py` в runtime-модуль и перевести `run_runtime_bot.py` на них.
    - changed files: `megabonk_bot/runtime/screen_state.py`, `run_runtime_bot.py`, `autopilot.py`, `tests/test_runtime_screen_state.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_screen_state.py tests/test_runtime_objective_cache.py tests/test_runtime_loop_recovery_flow.py -q`
    - limitations/follow-up: runtime больше не тянет screen-state helper напрямую из `autopilot.py`, но recovery/menu-click path всё ещё использует legacy `AutoPilot`; родительскую задачу держать открытой до дальнейшего сужения этой зависимости и живой проверки.

  - [x] Remove unused `AutoPilot` construction from `run_runtime_bot.py`; keep runtime screen detection on `RuntimeScreenDetector`.
    - changed files: `run_runtime_bot.py`, `tests/test_runtime_objective_cache.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: runtime entrypoint no longer imports or instantiates legacy `AutoPilot`, but recovery still uses template-click helpers and needs live-game validation before closing the parent `survival_mvp` task.
  - [x] Move recovery template-click helpers from `run_runtime_bot.py` into a dedicated runtime module.
    - changed files: `megabonk_bot/runtime/recovery_clicks.py`, `run_runtime_bot.py`, `tests/test_runtime_recovery_clicks.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_recovery_clicks.py tests/test_runtime_objective_cache.py tests/test_runtime_loop_recovery_flow.py -q`
    - limitations/follow-up: fallback menu clicks are now isolated from the entrypoint and unit-tested, but the parent `survival_mvp` task stays open until live-game validation confirms recovery behavior end to end.
  - [x] Move upgrade-dialog detection from `run_runtime_bot.py` into a dedicated runtime module.
    - changed files: `megabonk_bot/runtime/upgrade_dialog.py`, `run_runtime_bot.py`, `tests/test_runtime_upgrade_dialog.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_upgrade_dialog.py tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: upgrade detection is now isolated and covered by unit tests, but the parent `survival_mvp` task stays open until live-game validation confirms the remaining runtime path end to end.
  - [x] Move overlay-window helpers from `run_runtime_bot.py` into a dedicated runtime module.
    - changed files: `megabonk_bot/runtime/overlay_window.py`, `run_runtime_bot.py`, `tests/test_runtime_overlay_window.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_overlay_window.py tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: entrypoint no longer owns WinAPI overlay-window helpers, but the parent `survival_mvp` task stays open until live-game validation confirms the remaining runtime path end to end.
  - [x] Force `survival_only` runtime path to disable minimap, projectiles, world objects, and memory probe even if config flags regress.
    - changed files: `run_runtime_bot.py`, `tests/test_runtime_objective_cache.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: runtime now enforces the survival-only hot path against conflicting detection flags, but the parent `survival_mvp` task stays open until live-game validation confirms the remaining behavior end to end.
  - [x] Force `survival_only` runtime path to disable bunny-hop and slide even if config flags regress.
    - changed files: `run_runtime_bot.py`, `tests/test_runtime_objective_cache.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_objective_cache.py -q`
    - limitations/follow-up: navigation planner now receives survival-safe movement flags from runtime even if `max_policy` regresses, but the parent `survival_mvp` task stays open until live-game validation confirms the remaining behavior end to end.
  - changed files: `tests/test_runtime_objective_cache.py`, `tech_debt.md`
  - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_runtime_objective_cache.py tests/test_config.py -q`
  - limitations/follow-up: по репозиторию критерий выполнен: дефолтный launch использует `config/survival_mvp.yaml`, а runtime-guards принудительно отключают побочные survival-incompatible ветки; живая проверка в игре остаётся отдельным follow-up, но не блокирует закрытие этого пункта.

- [ ] Перевести активный профиль на рабочие каталоги ручных скринов.
  - Выбрать один формат: либо YAML как источник истины, либо сгенерированный JSON из YAML.
  - Активный `config/bot_profile.yaml` должен ссылаться на каталоги, где preview-файлы реально существуют.
  - Критерий готовности: `load_curated_catalogs()` показывает preview samples для всех записей survival-каталога.
  - [x] Переключить активные `bot_profile` и `survival_mvp` enemy/world catalog paths на YAML-манифесты с живыми preview-файлами.
    - changed files: `config/bot_profile.yaml`, `config/survival_mvp.yaml`, `tests/test_config.py`, `tests/test_asset_catalog.py`, `tech_debt.md`
    - tests/checks run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_asset_catalog.py -q`
    - limitations/follow-up: активные профили теперь явно используют `config/enemy_catalog.yaml` и `config/world_catalog.yaml`, а тест проверяет preview samples для всех текущих YAML entry; родительскую задачу оставить открытой, пока не решён единый формат для projectile-каталога и полного survival catalog workflow.

- [ ] Почистить каталоги от записей с отсутствующими файлами.
  - Сейчас JSON-каталоги содержат много ссылок на несуществующие `enemies/*.png`, `meshes/*.obj`, `silhouettes/*.png`, `minimap_icons/*.png`.
  - Для MVP оставить только сущности с реальными PNG/JPG примерами.
  - Критерий готовности: отдельная проверка валидирует, что каждый `preview_relpath`, `preview_relpaths`, `minimap_icon_relpath`, `silhouette_relpath` либо существует, либо отсутствует намеренно.

- [ ] Собрать минимальный датасет скринов врагов.
  - Начать с 3-5 типов врагов или хотя бы с орков в разных ракурсах, размерах и освещении.
  - Структура: `art_refs/megabonk_unity_extracts/imported/enemies/<enemy_id>/*.png`.
  - Критерий готовности: для каждого врага есть 5-15 живых gameplay-кропов, а не только Unity/иконки.

- [ ] Собрать минимальный датасет скринов декораций и препятствий.
  - Начать с деревьев, камней/стен, сундуков как нецели, краев/провалов если они видны на кадре.
  - Структура: `art_refs/megabonk_unity_extracts/imported/decor/<decor_id>/*.png` и `.../props/<prop_id>/*.png`.
  - Критерий готовности: navigation planner может отличать врага от крупной декорации и не шарахаться от каждого зеленого участка.

- [ ] Добавить smoke-проверку качества enemy detection на `screen.png` и нескольких ручных кадрах.
  - Проверять количество боксов, максимальную площадь бокса и отсутствие боксов, покрывающих большую часть экрана.
  - Критерий готовности: кадр с текущим ложным большим HSV-боксом больше не считается корректным enemy detection.

- [ ] Ограничить HSV enemy proposals.
  - Добавить фильтры по максимальной относительной площади, aspect ratio, нижней HUD/terrain-зоне и минимальной компактности контура.
  - Критерий готовности: зеленая земля/декорации не становятся врагом с rect почти во всю ширину экрана.

- [ ] Ввести confidence gate для классификации врагов.
  - Если proposal слишком большой, слишком широкий или плохо похож на каталог, не повышать его до `enemy_classes`.
  - Критерий готовности: `enemy_classes` не превращает любой большой зеленый фрагмент в `Orc` со score `1.0`.

- [ ] Добавить сглаживание угроз по времени.
  - Держать threat map за 3-5 последних кадров.
  - Направление менять только при устойчивом перевесе угрозы, а не каждый tick.
  - Критерий готовности: action reason и dir_id не скачут хаотично при одиночном шумовом боксе.

## P1 - Движение и поведение

- [ ] Сделать простую survival policy поверх надежных угроз.
  - Если центр опасен, strafe в сторону с меньшей угрозой.
  - Если опасность слева, двигаться вправо; если справа, влево; если угроз нет, идти вперед.
  - Прыжки и slide выключить по умолчанию до надежной оценки препятствий.
  - Критерий готовности: policy можно протестировать unit-тестами без игры.

- [ ] Уменьшить влияние camera yaw.
  - Не дергать мышь каждый tick.
  - Менять yaw только при устойчивой боковой угрозе или если бот застрял.
  - Критерий готовности: в логах нет постоянного чередования yaw `0/2` при слабом шуме.

- [ ] Разделить "угроза врага" и "препятствие".
  - Враги влияют на направление уклонения.
  - Декорации/стены влияют на запрет движения в полосу.
  - Критерий готовности: дерево не считается врагом, но может блокировать lane.

- [ ] Временно выключить сложные цели в `bot_profile.yaml`.
  - `projectiles_enabled: false`, `world_objects_enabled: false`, `minimap_enabled: false`, `allow_map_scan_tab: false`.
  - Не включать `collect_shrines_and_statues`, chest/open logic и boss prep в survival MVP.
  - Критерий готовности: runtime не открывает сундуки и не сканирует карту в MVP.

- [ ] Добавить runtime overlay/debug для survival decisions.
  - Показывать top threat boxes, выбранную safe lane, action reason, confidence.
  - Критерий готовности: по одному скриншоту overlay понятно, почему бот выбрал направление.

- [ ] Добавить short-run telemetry checklist.
  - После 1-2 минут игры проверять: средний FPS/tick, число enemy boxes, action reason distribution, death/recovery events.
  - Критерий готовности: по JSONL можно понять, бот умер из-за распознавания, движения или recovery.

## P2 - Удаление и упрощение

- [ ] Удалить или вынести из основного проекта старый RL-контур.
  - Кандидаты: `megabonk_env.py`, `train.py`, `play.py`, `megabonk_bot/env/`, тесты `tests/test_env_*`, `tests/test_megabonk_env_*`.
  - Перед удалением проверить `rg "MegabonkEnv|megabonk_env|train.py|play.py"`.
  - Критерий готовности: survival runtime запускается и тесты проходят без RL-зависимостей `gymnasium`, `stable-baselines3`, `torch`.

- [ ] Удалить tracked TensorBoard artifact.
  - Кандидат: `tb/PPO_1/events.out.tfevents.*`.
  - Добавить `tb/` в `.gitignore`.
  - Критерий готовности: `git ls-files tb` пустой, новые TensorBoard-логи не попадают в git.

- [ ] Удалить или заменить устаревший `art_refs/megabonk_unity_extracts/report.json`.
  - Сейчас report заявляет `minimap_icons`, `silhouettes`, `meshes`, но этих папок нет.
  - Варианты: удалить report как stale artifact или регенерировать после реального extraction.
  - Критерий готовности: report не описывает отсутствующие файлы.

- [ ] Разобраться с корневыми дублями модулей.
  - Кандидаты: `regions.py`, `vision.py` при наличии `megabonk_bot/regions.py` и `megabonk_bot/vision.py`.
  - Перед удалением проверить импорты и документацию.
  - Критерий готовности: нет двух источников истины для regions/vision.

- [ ] Сократить зависимость runtime от `autopilot.py`.
  - Сейчас `run_runtime_bot.py` импортирует `AutoPilot` и `is_death_like_frame`.
  - Перенести screen-state/death helpers в пакет `megabonk_bot/runtime` или оставить `autopilot.py` только как legacy.
  - Критерий готовности: runtime entrypoint не зависит от экспериментального autopilot-файла.

- [ ] Разбить `run_runtime_bot.py`.
  - Кандидаты на вынос: CLI/config bootstrap, window/overlay WinAPI helpers, runtime tick, recovery, catalog loading.
  - Критерий готовности: entrypoint остается тонким, основные компоненты тестируются отдельно.

- [ ] Разбить `megabonk_bot/recognition.py`.
  - Кандидаты на вынос: enemy detection, catalog classification, minimap, overlay drawing, HUD drawing.
  - Критерий готовности: survival detection можно менять без риска сломать minimap/overlay.

- [ ] Разбить `megabonk_bot/hud.py` или изолировать fast timer OCR.
  - Файл уже большой и содержит несколько независимых подходов к OCR.
  - Критерий готовности: HUD не мешает survival MVP и может быть выключен без побочных эффектов.

- [ ] Почистить ignored локальные артефакты.
  - Кандидаты: `logs/`, `dbg/`, `dbg_hud/`, `.pytest_cache/`, `__pycache__/`.
  - Не удалять ручные скрины, если они используются каталогами; сначала перенести их из `dbg_hud` в `art_refs/.../imported`.
  - Критерий готовности: рабочая копия не хранит временные файлы как источник истины.

- [ ] Расширить `.gitignore`.
  - Добавить `.venv/`, `.pytest_cache/`, `tb/`, runtime JSONL/logs, debug dumps, временные crop/export директории.
  - Отдельно решить судьбу `screen.png`: либо оставить как тестовый fixture, либо перенести в `tests/fixtures/`.
  - Критерий готовности: `git status --short` не засоряется локальными runtime-артефактами.

## P3 - Инструменты и воспроизводимость

- [ ] Добавить проверку целостности каталогов.
  - Команда должна валидировать все active catalog paths из `config/bot_profile.yaml`.
  - Проверять существование файлов, число previews, пустые catalogs, дубли entity_id.
  - Критерий готовности: перед запуском runtime понятно, какие ассеты не найдены.

- [ ] Довести `scripts/curate_catalog.py` до основного workflow.
  - Документировать, как drag-and-drop скрины врагов/декораций попадают в `art_refs/.../imported`.
  - Добавить неинтерактивный CLI режим для batch-добавления.
  - Критерий готовности: новые скрины можно добавить без ручного редактирования YAML/JSON.

- [ ] Зафиксировать зависимости проекта.
  - Создать `requirements.txt` или `pyproject.toml`.
  - Разделить runtime, dev/test и legacy/RL зависимости.
  - Критерий готовности: проект поднимается на чистой машине одной командой установки.

- [ ] Добавить команду smoke-run без игры.
  - Прогонять `screen.png` и fixture-кадры через detection -> snapshot -> policy.
  - Выводить summary: boxes, threats, action, reasons.
  - Критерий готовности: можно быстро проверить, что survival behavior не развалился.

- [ ] Обновить README после стабилизации MVP.
  - Убрать акцент на все-и-сразу.
  - Описать один рекомендуемый запуск survival MVP, сбор скринов и проверку каталогов.
  - Критерий готовности: новый пользователь не запускает случайно старый RL или max-policy путь.

## Не делать сейчас

- Не включать сбор лута, сундуки, shrine и projectiles до стабильного распознавания врагов.
- Не опираться на Unity extraction как на основной источник gameplay-распознавания, пока `UnityPy` не установлен и extraction не дает реальные PNG.
- Не удалять `dbg_hud/*` до переноса полезных ручных скринов в `art_refs/.../imported`.
- Не расширять memory probe, пока survival MVP не работает на одном кадре и простом runtime-цикле.
