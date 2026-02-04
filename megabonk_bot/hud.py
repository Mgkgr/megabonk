import importlib.util
import logging
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

_TESSERACT_LAST_ERROR_AT = None
_TESSERACT_LAST_WARNED_AT = 0.0
_TESSERACT_CONFIGURED = False
_TESSERACT_WARNING_INTERVAL = 60.0
_TESSERACT_RETRY_INTERVAL = 30.0

DEFAULT_HUD_REGIONS = {
    "hp": (0.04, 0.02, 0.12, 0.05),
    "gold": (0.85, 0.02, 0.12, 0.05),
    "time": (0.39, 0.012, 0.22, 0.06),
}


def _get_tesseract():
    global _TESSERACT_CONFIGURED, _TESSERACT_LAST_ERROR_AT
    if _TESSERACT_LAST_ERROR_AT is not None:
        if time.monotonic() - _TESSERACT_LAST_ERROR_AT < _TESSERACT_RETRY_INTERVAL:
            return None
    if importlib.util.find_spec("pytesseract") is None:
        _TESSERACT_LAST_ERROR_AT = time.monotonic()
        return None
    import pytesseract
    if not _TESSERACT_CONFIGURED:
        cmd_from_env = os.getenv("TESSERACT_CMD")
        if cmd_from_env:
            pytesseract.pytesseract.tesseract_cmd = cmd_from_env
        elif os.name == "nt":
            default_cmd = Path(
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            if default_cmd.exists():
                pytesseract.pytesseract.tesseract_cmd = str(default_cmd)
        _TESSERACT_CONFIGURED = True
    _TESSERACT_LAST_ERROR_AT = None
    return pytesseract


def _warn_missing_tesseract():
    global _TESSERACT_LAST_WARNED_AT
    now = time.monotonic()
    if now - _TESSERACT_LAST_WARNED_AT >= _TESSERACT_WARNING_INTERVAL:
        LOGGER.warning("Tesseract OCR не найден — проверьте установку и PATH")
        _TESSERACT_LAST_WARNED_AT = now


def _resolve_region(frame, regions, key):
    if regions:
        reg_key = f"REG_HUD_{key.upper()}"
        if reg_key in regions:
            return regions[reg_key]
    h, w = frame.shape[:2]
    rx, ry, rw, rh = DEFAULT_HUD_REGIONS[key]
    return (int(rx * w), int(ry * h), int(rw * w), int(rh * h))


def _preprocess_for_ocr(roi_bgr, *, scale=3.0, adaptive=False):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    if adaptive:
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            2,
        )
    else:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return thresh


def _ocr_text(image, whitelist, psm=7):
    global _TESSERACT_LAST_ERROR_AT
    pytesseract = _get_tesseract()
    if pytesseract is None:
        return None, None
    config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        _TESSERACT_LAST_ERROR_AT = time.monotonic()
        _warn_missing_tesseract()
        return None, None
    if not data.get("text"):
        return None, None
    best_idx = None
    best_conf = -1.0
    for idx, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][idx])
        except (TypeError, ValueError):
            continue
        if conf > best_conf:
            best_conf = conf
            best_idx = idx
    if best_idx is None:
        try:
            raw = pytesseract.image_to_string(image, config=config).strip()
        except pytesseract.TesseractNotFoundError:
            _TESSERACT_LAST_ERROR_AT = time.monotonic()
            _warn_missing_tesseract()
            return None, None
        if not raw:
            return None, None
        return raw, 0.0
    return data["text"][best_idx].strip(), best_conf


def _iter_preprocessed(roi_bgr):
    for scale in (2.5, 3.0, 3.5, 4.0):
        yield _preprocess_for_ocr(roi_bgr, scale=scale, adaptive=False)
        yield _preprocess_for_ocr(roi_bgr, scale=scale, adaptive=True)


def _best_ocr(roi_bgr, whitelist):
    best_text = None
    best_conf = None
    for prep in _iter_preprocessed(roi_bgr):
        for psm in (7, 6):
            text, conf = _ocr_text(prep, whitelist=whitelist, psm=psm)
            if text is None or conf is None:
                continue
            if best_conf is None or conf > best_conf:
                best_conf = conf
                best_text = text
    return best_text, best_conf


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


def read_hud_values(frame_bgr, regions=None, min_conf=45.0):
    if frame_bgr is None or frame_bgr.size == 0:
        return {"hp": None, "gold": None, "time": None}

    pytesseract = _get_tesseract()
    if pytesseract is None:
        if _TESSERACT_LAST_ERROR_AT is not None:
            _warn_missing_tesseract()
        else:
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
        text, conf = _best_ocr(roi, whitelist=whitelist)
        if key == "time":
            value = _parse_time(text) if text else None
            if value is not None:
                results[key] = value
                continue
            key_min_conf = min_conf * 0.5
            key_soft_conf = min_conf * 0.25
        else:
            key_min_conf = min_conf
            key_soft_conf = min_conf * 0.6
        if text is None or conf is None or conf < key_min_conf:
            if text is None or conf is None or conf < key_soft_conf:
                results[key] = None
                continue
        if key == "time":
            value = _parse_time(text)
        else:
            value = _parse_int(text)
        results[key] = value
    return results
