from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CategoryPreset:
    category: str
    kind: str
    rel_dir: str
    default_poi_type: str | None = None


@dataclass(frozen=True)
class CurateResult:
    category: str
    entity_id: str
    manifest_path: Path
    asset_refs_dir: Path
    copied_files: tuple[Path, ...]
    preview_relpaths: tuple[str, ...]


CATEGORY_PRESETS: dict[str, CategoryPreset] = {
    "enemies": CategoryPreset(category="enemies", kind="enemy", rel_dir="imported/enemies"),
    "props": CategoryPreset(category="props", kind="poi", rel_dir="imported/props"),
    "decor": CategoryPreset(category="decor", kind="poi", rel_dir="imported/decor", default_poi_type="decor"),
}


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "item"


def parse_dropped_paths(text: str) -> list[str]:
    if not text:
        return []
    output: list[str] = []
    for line in str(text).replace("\r", "\n").splitlines():
        current = line.strip()
        if not current:
            continue
        try:
            tokens = shlex.split(current, posix=False)
        except ValueError:
            tokens = [current]
        for token in tokens:
            item = str(token).strip().strip('"').strip("'")
            if item:
                output.append(item)
    return output


def _ordered_unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(item)
    return output


def _read_manifest_payload(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Для YAML-манифеста нужен пакет pyyaml (`pip install pyyaml`).") from exc
        payload = yaml.safe_load(raw) or []
    else:
        payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError(f"Catalog manifest must be a list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _write_manifest_payload(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Для YAML-манифеста нужен пакет pyyaml (`pip install pyyaml`).") from exc
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_samples(
    input_paths: Iterable[str | Path],
    *,
    asset_refs_dir: Path,
    entity_id: str,
    rel_dir: str,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    target_dir = asset_refs_dir / rel_dir / entity_id
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []
    preview_relpaths: list[str] = []
    for index, raw_path in enumerate(input_paths, start=1):
        src = Path(raw_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Screenshot not found: {src}")
        ext = src.suffix.lower() or ".png"
        dst_name = f"{index:02d}_{_slug(src.stem)}{ext}"
        dst = target_dir / dst_name
        shutil.copy2(src, dst)
        copied_files.append(dst)
        preview_relpaths.append(dst.relative_to(asset_refs_dir).as_posix())
    if not copied_files:
        raise ValueError("At least one screenshot path is required")
    return tuple(copied_files), tuple(preview_relpaths)


def curate_samples(
    *,
    category: str,
    entity_id: str,
    display_name: str,
    input_paths: Iterable[str | Path],
    manifest_path: str | Path,
    asset_refs_dir: str | Path,
    aliases: Iterable[str] = (),
    family: str | None = None,
    poi_type: str | None = None,
    kind: str | None = None,
) -> CurateResult:
    normalized_category = str(category).strip().lower()
    if normalized_category not in CATEGORY_PRESETS:
        raise ValueError(f"Unsupported category: {category}")
    preset = CATEGORY_PRESETS[normalized_category]
    normalized_entity_id = _slug(entity_id)
    manifest = Path(manifest_path).resolve()
    asset_root = Path(asset_refs_dir).resolve()
    copied_files, preview_relpaths = _copy_samples(
        input_paths,
        asset_refs_dir=asset_root,
        entity_id=normalized_entity_id,
        rel_dir=preset.rel_dir,
    )

    resolved_aliases = _ordered_unique(aliases)
    resolved_family = str(family).strip() if family else normalized_entity_id
    resolved_poi_type = str(poi_type).strip() if poi_type else preset.default_poi_type
    entry: dict[str, Any] = {
        "entity_id": normalized_entity_id,
        "kind": str(kind).strip() if kind else preset.kind,
        "display_name": str(display_name).strip() or normalized_entity_id.replace("_", " ").title(),
        "aliases": resolved_aliases,
        "preview_relpath": preview_relpaths[0],
        "preview_relpaths": list(preview_relpaths),
        "metadata": {
            "family": resolved_family,
            "variant": "default",
        },
    }
    if resolved_poi_type:
        entry["poi_type"] = resolved_poi_type

    payload = _read_manifest_payload(manifest)
    updated = False
    for index, current in enumerate(payload):
        if str(current.get("entity_id", "")).strip().lower() == normalized_entity_id:
            payload[index] = entry
            updated = True
            break
    if not updated:
        payload.append(entry)
    payload.sort(key=lambda current: str(current.get("entity_id", "")).lower())
    _write_manifest_payload(manifest, payload)

    return CurateResult(
        category=normalized_category,
        entity_id=normalized_entity_id,
        manifest_path=manifest,
        asset_refs_dir=asset_root,
        copied_files=copied_files,
        preview_relpaths=preview_relpaths,
    )


def run_interactive_curator(
    *,
    project_root: str | Path,
    input_func=input,
    print_func=print,
) -> CurateResult:
    root = Path(project_root).resolve()
    asset_refs_default = root / "art_refs" / "megabonk_unity_extracts"
    if not asset_refs_default.exists():
        asset_refs_default = root / "art_refs"

    print_func("Выбери тип:")
    print_func("1. decor - декорации")
    print_func("2. props - сундуки/статуи")
    print_func("3. enemies - враги")
    selection = str(input_func("Номер или имя типа: ")).strip().lower()
    mapping = {"1": "decor", "2": "props", "3": "enemies", "decor": "decor", "props": "props", "enemies": "enemies"}
    category = mapping.get(selection)
    if category is None:
        raise ValueError(f"Unknown selection: {selection}")

    entity_id = str(input_func("entity_id: ")).strip()
    display_name = str(input_func("display_name (Enter = auto): ")).strip()
    aliases = str(input_func("aliases через запятую (можно пусто): ")).strip()

    family = None
    poi_type = None
    if category in {"props", "decor"}:
        family = str(input_func("family (например chest/statue/shrine/decor): ")).strip()
        poi_type = str(input_func("poi_type (например loot/shrine/decor, можно пусто): ")).strip()

    default_manifest = root / "config" / ("enemy_catalog.yaml" if category == "enemies" else "world_catalog.yaml")
    manifest_raw = str(input_func(f"manifest path (Enter = {default_manifest}): ")).strip()
    manifest_path = Path(manifest_raw).expanduser() if manifest_raw else default_manifest

    asset_refs_raw = str(input_func(f"asset_refs_dir (Enter = {asset_refs_default}): ")).strip()
    asset_refs_dir = Path(asset_refs_raw).expanduser() if asset_refs_raw else asset_refs_default

    print_func("Вставь пути к PNG/JPG. Можно drag-and-drop одной строкой. Пустая строка завершает ввод.")
    collected_paths: list[str] = []
    while True:
        line = str(input_func("> ")).strip()
        if not line:
            break
        collected_paths.extend(parse_dropped_paths(line))

    alias_items = [item.strip() for item in aliases.split(",")] if aliases else []
    result = curate_samples(
        category=category,
        entity_id=entity_id,
        display_name=display_name or entity_id.replace("_", " ").title(),
        input_paths=collected_paths,
        manifest_path=manifest_path if manifest_path.is_absolute() else root / manifest_path,
        asset_refs_dir=asset_refs_dir if asset_refs_dir.is_absolute() else root / asset_refs_dir,
        aliases=alias_items,
        family=family or None,
        poi_type=poi_type or None,
    )
    print_func(f"Готово: {result.manifest_path}")
    print_func(f"Скопировано файлов: {len(result.copied_files)}")
    return result
