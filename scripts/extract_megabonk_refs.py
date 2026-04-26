#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = (
    "Minimap",
    "Objective",
    "Interactable",
    "Projectile",
    "Chest",
    "Shrine",
    "Crypt",
    "GhostGrave",
    "Spike",
    "Boss",
    "Portal",
    "Key",
    "t_",
    "FBX_",
)


CURATED_OCR_NORMALIZE = {
    "objective arrow": "objective",
    "first objective": "objective",
    "challenge shrine": "challenge_shrine",
    "balance shrine": "balance_shrine",
    "charge shrine": "charge_shrine",
    "charge upgrade": "charge_shrine",
    "cursed shrine": "cursed_shrine",
    "greedy gold": "greed_shrine",
    "greed shrine": "greed_shrine",
    "magnet shrine": "magnet_shrine",
    "moai shrine": "moai_shrine",
    "portal final": "interactable_portal_final",
    "boss portal": "interactable_portal",
    "crypt exit": "interactable_crypt_leave",
    "crypt key": "crypt_key",
    "cage key": "interactable_cage_key",
    "boss marker": "boss_marker",
    "graveyard boss portal": "graveyard_boss_portal",
    "graveyard boss metal gate": "graveyard_boss_metal_gate",
    "ghost grave": "ghost_grave",
    "desert grave": "interactable_desert_grave",
    "boss lamp": "boss_lamp",
    "escape first crypt": "challenge_escape_first_crypt",
    "kill tier boss": "challenge_kill_tier_boss",
}


METADATA_KEYWORDS = (
    "Minimap",
    "Objective",
    "Interactable",
    "Projectile",
    "Chest",
    "Shrine",
    "Crypt",
    "Ghost",
    "Spike",
    "Portal",
    "Boss",
    "Key",
    "ChallengeModifier",
)


@dataclass(frozen=True)
class ExtractedAsset:
    name: str
    kind: str
    source_file: str
    output_relpath: str


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("_") or "unnamed"


def _humanize_identifier(value: str) -> str:
    text = str(value).replace("_", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _resolve_data_dir(path: Path) -> Path:
    if (path / "il2cpp_data").exists():
        return path
    candidate = path / "Megabonk_Data"
    if (candidate / "il2cpp_data").exists():
        return candidate
    raise FileNotFoundError(f"Megabonk_Data not found under {path}")


def _read_metadata_strings(metadata_path: Path) -> list[str]:
    raw = metadata_path.read_bytes()
    hits = re.findall(rb"[A-Za-z0-9_./:-]{4,}", raw)
    values = sorted({item.decode("utf-8", errors="ignore") for item in hits})
    return [item for item in values if any(keyword.lower() in item.lower() for keyword in METADATA_KEYWORDS)]


def _read_catalog_terms(streaming_assets_dir: Path) -> list[str]:
    catalog_path = streaming_assets_dir / "aa" / "catalog.json"
    if not catalog_path.exists():
        return []
    text = catalog_path.read_text(encoding="utf-8", errors="ignore")
    words = re.findall(r"[A-Za-z0-9_./:-]{4,}", text)
    output = sorted(
        {
            word
            for word in words
            if any(keyword.lower() in word.lower() for keyword in ("localization", "objective", "challenge", "shrine", "crypt", "boss", "portal"))
        }
    )
    return output


def _build_ocr_lexicon(metadata_terms: list[str], catalog_terms: list[str]) -> dict[str, Any]:
    token_seed = {
        "objective",
        "objective arrow",
        "first objective",
        "challenge",
        "challenge shrine",
        "balance shrine",
        "charge shrine",
        "charge upgrade",
        "cursed shrine",
        "greed shrine",
        "greedy gold",
        "magnet shrine",
        "moai shrine",
        "portal",
        "boss portal",
        "crypt",
        "crypt exit",
        "crypt key",
        "graveyard boss portal",
        "graveyard boss metal gate",
        "boss lamp",
        "boss marker",
        "chest",
        "chest free",
        "chest ghost",
        "chest evil",
        "cage key",
    }
    phrase_seed = {
        item
        for item in metadata_terms
        if any(keyword.lower() in item.lower() for keyword in ("Shrine", "Chest", "Portal", "Objective", "Crypt", "Boss", "Key", "ChallengeModifier"))
    }
    phrase_seed.update(catalog_terms)
    token_seed.update(_humanize_identifier(item) for item in phrase_seed)
    token_seed.update(_humanize_identifier(item) for item in metadata_terms if item.startswith("SHRINE_"))
    tokens = sorted({item for item in token_seed if item})
    phrases = sorted({item for item in phrase_seed if item})
    return {
        "tokens": tokens,
        "phrases": phrases,
        "normalize": dict(sorted(CURATED_OCR_NORMALIZE.items())),
    }


def _ensure_cv2():
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    return cv2, np


def _write_reference_icons(output_dir: Path) -> list[str]:
    cv2, np = _ensure_cv2()
    icon_dir = output_dir / "minimap_icons"
    icon_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "boss_marker.png": ("circle", (0, 0, 255)),
        "portal.png": ("diamond", (255, 0, 0)),
        "shrine.png": ("hex", (0, 200, 255)),
        "chest.png": ("square", (0, 128, 255)),
        "key.png": ("key", (255, 255, 0)),
        "objective.png": ("diamond", (255, 255, 255)),
        "crypt_exit.png": ("diamond", (255, 0, 255)),
        "player.png": ("arrow", (255, 255, 255)),
    }

    written: list[str] = []
    for filename, (shape, color) in specs.items():
        canvas = np.zeros((20, 20, 3), dtype=np.uint8)
        if shape == "circle":
            cv2.circle(canvas, (10, 10), 5, color, -1)
        elif shape == "diamond":
            pts = np.array([[10, 3], [16, 10], [10, 17], [4, 10]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, color)
        elif shape == "hex":
            pts = np.array([[10, 2], [16, 6], [16, 14], [10, 18], [4, 14], [4, 6]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, color)
        elif shape == "square":
            cv2.rectangle(canvas, (5, 5), (15, 15), color, -1)
        elif shape == "key":
            cv2.circle(canvas, (7, 10), 3, color, -1)
            cv2.rectangle(canvas, (10, 9), (16, 11), color, -1)
            cv2.rectangle(canvas, (13, 11), (14, 15), color, -1)
            cv2.rectangle(canvas, (15, 11), (16, 13), color, -1)
        elif shape == "arrow":
            pts = np.array([[10, 3], [15, 14], [10, 11], [5, 14]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, color)
        else:
            continue
        path = icon_dir / filename
        cv2.imwrite(str(path), canvas)
        written.append(str(path.relative_to(output_dir)).replace("\\", "/"))
    return written


def _parse_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            parts = line.split()[1:]
            face: list[int] = []
            for part in parts:
                token = part.split("/", 1)[0]
                if not token:
                    continue
                index = int(token)
                if index < 0:
                    index = len(vertices) + index + 1
                face.append(index - 1)
            if len(face) >= 3:
                faces.append(face)
    return vertices, faces


def _render_projection(vertices, faces, axes, *, size: int = 80):
    cv2, np = _ensure_cv2()
    pts = np.array([[vertex[axes[0]], vertex[axes[1]]] for vertex in vertices], dtype=np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (pts - mins) / span
    margin = 6.0
    scale = float(size - (margin * 2))
    norm[:, 0] = margin + norm[:, 0] * scale
    norm[:, 1] = margin + (1.0 - norm[:, 1]) * scale
    canvas = np.zeros((size, size), dtype=np.uint8)
    if faces:
        for face in faces:
            polygon = np.array([norm[index] for index in face if 0 <= index < len(norm)], dtype=np.int32)
            if len(polygon) >= 3:
                cv2.fillConvexPoly(canvas, polygon, 255)
    else:
        for point in norm.astype(np.int32):
            cv2.circle(canvas, tuple(int(item) for item in point), 1, 255, -1)
    return canvas


def _render_silhouettes(output_dir: Path) -> list[str]:
    cv2, np = _ensure_cv2()
    mesh_dir = output_dir / "meshes"
    if not mesh_dir.exists():
        return []
    silhouette_dir = output_dir / "silhouettes"
    silhouette_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for obj_path in sorted(mesh_dir.glob("*.obj")):
        vertices, faces = _parse_obj(obj_path)
        if not vertices:
            continue
        projections = {
            "xy": _render_projection(vertices, faces, (0, 1)),
            "xz": _render_projection(vertices, faces, (0, 2)),
            "yz": _render_projection(vertices, faces, (1, 2)),
        }
        best = max(projections.values(), key=lambda image: int(np.count_nonzero(image)))
        out_path = silhouette_dir / f"{obj_path.stem}.png"
        cv2.imwrite(str(out_path), best)
        written.append(str(out_path.relative_to(output_dir)).replace("\\", "/"))
    return written


def _classify_output_dir(asset_name: str) -> str:
    lowered = asset_name.lower()
    if "minimap" in lowered or "objective" in lowered:
        return "minimap_icons"
    if "tome" in lowered or "holybook" in lowered:
        return "tomes"
    return "enemies"


def _matches_patterns(name: str, patterns: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _extract_with_unitypy(game_data_dir: Path, output_dir: Path, patterns: tuple[str, ...]) -> tuple[list[ExtractedAsset], list[str]]:
    extracted: list[ExtractedAsset] = []
    errors: list[str] = []
    try:
        import UnityPy  # type: ignore
    except Exception:
        return extracted, ["UnityPy is not installed; skipped sharedassets extraction"]

    seen_outputs: set[str] = set()
    for asset_path in sorted(game_data_dir.glob("sharedassets*.assets")):
        try:
            env = UnityPy.load(str(asset_path))
        except Exception as exc:
            errors.append(f"{asset_path.name}: load failed: {exc}")
            continue
        for obj in env.objects:
            type_name = str(getattr(obj.type, "name", obj.type))
            if type_name not in {"Texture2D", "Sprite"}:
                continue
            try:
                data = obj.read()
                name = str(getattr(data, "name", "") or getattr(data, "m_Name", "") or "").strip()
                if not name or not _matches_patterns(name, patterns):
                    continue
                image = getattr(data, "image", None)
                if image is None:
                    continue
                target_dir = output_dir / _classify_output_dir(name)
                target_dir.mkdir(parents=True, exist_ok=True)
                relpath = f"{target_dir.relative_to(output_dir).as_posix()}/{_slug(name)}.png"
                if relpath in seen_outputs:
                    continue
                out_path = output_dir / relpath
                image.save(out_path)
                seen_outputs.add(relpath)
                extracted.append(
                    ExtractedAsset(
                        name=name,
                        kind=type_name,
                        source_file=asset_path.name,
                        output_relpath=relpath,
                    )
                )
            except Exception as exc:
                errors.append(f"{asset_path.name}: {type_name}: {exc}")
    return extracted, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Megabonk reference assets and helper atlases.")
    parser.add_argument("--game-dir", required=True, help="Path to Megabonk or Megabonk_Data")
    parser.add_argument("--output", default="art_refs/megabonk_unity_extracts", help="Output directory for extracted refs")
    parser.add_argument("--ocr-lexicon-out", default="", help="Optional output path for generated OCR lexicon JSON")
    parser.add_argument("--report-name", default="report.json", help="Report filename to write under output")
    parser.add_argument("--patterns", nargs="*", default=list(DEFAULT_PATTERNS), help="Substring patterns to extract from sharedassets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_dir = _resolve_data_dir(Path(args.game_dir).resolve())
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = game_dir / "il2cpp_data" / "Metadata" / "global-metadata.dat"
    streaming_assets_dir = game_dir / "StreamingAssets"

    metadata_terms = _read_metadata_strings(metadata_path) if metadata_path.exists() else []
    catalog_terms = _read_catalog_terms(streaming_assets_dir) if streaming_assets_dir.exists() else []
    ocr_lexicon = _build_ocr_lexicon(metadata_terms, catalog_terms)

    extracted_assets, extraction_errors = _extract_with_unitypy(game_dir, output_dir, tuple(args.patterns))
    icon_files = _write_reference_icons(output_dir)
    silhouette_files = _render_silhouettes(output_dir)

    if args.ocr_lexicon_out:
        lexicon_path = Path(args.ocr_lexicon_out).resolve()
        lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        lexicon_path.write_text(json.dumps(ocr_lexicon, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "game_data_dir": str(game_dir),
        "output_dir": str(output_dir),
        "patterns": list(args.patterns),
        "extracted_assets": [asdict(item) for item in extracted_assets],
        "metadata_terms_count": len(metadata_terms),
        "catalog_terms_count": len(catalog_terms),
        "ocr_lexicon": {
            "tokens_count": len(ocr_lexicon["tokens"]),
            "phrases_count": len(ocr_lexicon["phrases"]),
        },
        "generated_icon_files": icon_files,
        "generated_silhouettes": silhouette_files,
        "errors": extraction_errors,
    }
    report_path = output_dir / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report to {report_path}")
    if args.ocr_lexicon_out:
        print(f"Wrote OCR lexicon to {Path(args.ocr_lexicon_out).resolve()}")


if __name__ == "__main__":
    main()
