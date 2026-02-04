import importlib.util
import logging
import re

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_HUD_REGIONS = {
    "hp": (0.04, 0.02, 0.12, 0.05),
    "gold": (0.85, 0.02, 0.12, 0.05),
    "time": (0.45, 0.02, 0.10, 0.05),
}


def _get_tesseract():
    if importlib.util.find_spec("pytesseract") is None:
        return None
    import pytesseract

    return pytesseract


def _resolve_region(frame, regions, key):
    if regions:
        reg_key = f"REG_HUD_{key.upper()}"
        if reg_key in regions:
            return regions[reg_key]
    h, w = frame.shape[:2]
    rx, ry, rw, rh = DEFAULT_HUD_REGIONS[key]
    return (int(rx * w), int(ry * h), int(rw * w), int(rh * h))


def _preprocess_for_ocr(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    scale = 2.0
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def _ocr_text(image, whitelist, psm=7):
    pytesseract = _get_tesseract()
    if pytesseract is None:
        return None, None
    config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(
        image, config=config, output_type=pytesseract.Output.DICT
    )
    if not data.get("text"):
        return None, None
    best_idx = None
    best_conf = -1.0
    for idx, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = float(data["conf"][idx])
        if conf > best_conf:
            best_conf = conf
            best_idx = idx
    if best_idx is None:
        return None, None
    return data["text"][best_idx].strip(), best_conf


def _parse_int(text):
    digits = re.findall(r"\d+", text)
    if not digits:
        return None
    return int("".join(digits))


def _parse_time(text):
    cleaned = re.sub(r"[^0-9:]", "", text)
    if ":" in cleaned:
        parts = cleaned.split(":")
        if len(parts) >= 2:
            try:
                minutes = int(parts[-2])
                seconds = int(parts[-1])
            except ValueError:
                return None
            return minutes * 60 + seconds
    digits = re.findall(r"\d+", cleaned)
    if not digits:
        return None
    return int("".join(digits))


def read_hud_values(frame_bgr, regions=None, min_conf=55.0):
    if frame_bgr is None or frame_bgr.size == 0:
        return {"hp": None, "gold": None, "time": None}

    pytesseract = _get_tesseract()
    if pytesseract is None:
        LOGGER.debug("pytesseract не найден — HUD OCR отключён")
        return {"hp": None, "gold": None, "time": None}

    results = {}
    for key, whitelist in (
        ("hp", "0123456789"),
        ("gold", "0123456789"),
        ("time", "0123456789:"),
    ):
        x, y, w, h = _resolve_region(frame_bgr, regions, key)
        roi = frame_bgr[y : y + h, x : x + w]
        if roi.size == 0:
            results[key] = None
            continue
        prep = _preprocess_for_ocr(roi)
        text, conf = _ocr_text(prep, whitelist=whitelist, psm=7)
        if text is None or conf is None or conf < min_conf:
            results[key] = None
            continue
        if key == "time":
            results[key] = _parse_time(text)
        else:
            results[key] = _parse_int(text)
    return results
