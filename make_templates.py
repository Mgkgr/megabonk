from megabonk_bot.dpi import enable_dpi_awareness

enable_dpi_awareness()

import argparse
import json
from pathlib import Path

import cv2

from megabonk_bot.hud import resolve_hud_debug_rects
from megabonk_bot.regions import build_regions

DEFAULT_SRC = "screen.png"
DEFAULT_TEMPLATE_DIR = "templates"
DEFAULT_HUD_OUTDIR = "dbg_hud/screen_crops"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Template cropper and HUD ROI exporter.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SRC,
        help="Path to source screenshot.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_TEMPLATE_DIR,
        help="Output directory for template crops.",
    )
    parser.add_argument(
        "--export-hud",
        action="store_true",
        help="Export HUD OCR fields (time/kills/lvl/gold/hp) from screenshot.",
    )
    parser.add_argument(
        "--hud-outdir",
        default=DEFAULT_HUD_OUTDIR,
        help="Output directory for HUD field crops.",
    )
    return parser.parse_args()


def _load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Не нашёл изображение: {path}")
    return img


def _interactive_crop_template(img, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    roi = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise SystemExit("ROI не выбрана")
    crop = img[y : y + h, x : x + w]
    name = input("Имя шаблона (например tpl_play): ").strip()
    if not name:
        raise SystemExit("Пустое имя шаблона")
    out_path = outdir / f"{name}.png"
    cv2.imwrite(str(out_path), crop)
    print(f"Saved: {out_path}")
    cv2.destroyAllWindows()


def _export_hud_rois(img, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    regions = build_regions(w, h)
    rects = resolve_hud_debug_rects(img, regions=regions)
    if not rects:
        raise SystemExit("Не удалось вычислить ROI HUD для текущего скрина")

    preview = img.copy()
    meta = {}
    for key in ("time", "kills", "lvl", "gold", "hp"):
        rect = rects.get(key)
        if rect is None:
            continue
        x, y, rw, rh = rect
        crop = img[y : y + rh, x : x + rw]
        if crop.size == 0:
            continue
        filename = f"{key}.png"
        out_path = outdir / filename
        cv2.imwrite(str(out_path), crop)
        meta[key] = {
            "rect": [int(x), int(y), int(rw), int(rh)],
            "file": filename,
        }
        cv2.rectangle(preview, (x, y), (x + rw, y + rh), (0, 255, 255), 2)
        cv2.putText(
            preview,
            f"{key} ({x},{y},{rw},{rh})",
            (x, max(14, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    preview_path = outdir / "hud_preview.png"
    cv2.imwrite(str(preview_path), preview)
    meta_path = outdir / "hud_regions.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved HUD crops: {outdir}")
    print(f"Saved HUD preview: {preview_path}")
    print(f"Saved HUD regions: {meta_path}")


def main():
    args = _parse_args()
    source = Path(args.source)
    img = _load_image(source)
    if args.export_hud:
        _export_hud_rois(img, Path(args.hud_outdir))
    else:
        _interactive_crop_template(img, Path(args.outdir))


if __name__ == "__main__":
    main()
