import cv2
import numpy as np
from pathlib import Path


def match_template(gray_img, gray_tpl):
    res = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCOEFF_NORMED)
    minv, maxv, minp, maxp = cv2.minMaxLoc(res)
    return maxv, maxp  # score, top-left


def _to_gray(img):
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (bgr * alpha).astype(np.uint8)
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def find_in_region(
    frame_bgr,
    tpl_bgr,
    region,
    threshold=0.80,
    scales=(0.75, 0.85, 0.95, 1.00, 1.10, 1.25, 1.40, 1.60),
    method=cv2.TM_CCOEFF_NORMED,
):
    """
    region: (x,y,w,h) в координатах кадра
    Возвращает (found, (cx,cy), score)
    """
    x, y, w, h = region
    roi = frame_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return False, (0, 0), 0.0

    g_roi = _to_gray(roi)
    g_tpl0 = _to_gray(tpl_bgr)
    if g_tpl0 is None or g_tpl0.size == 0:
        return False, (0, 0), 0.0

    best_score = -1.0
    best_center = (0, 0)
    h0, w0 = g_tpl0.shape[:2]

    for scale in scales:
        tw = int(w0 * scale)
        th = int(h0 * scale)
        if tw < 8 or th < 8:
            continue
        if tw >= g_roi.shape[1] or th >= g_roi.shape[0]:
            continue

        g_tpl = cv2.resize(g_tpl0, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(g_roi, g_tpl, method)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv > best_score:
            cx = x + maxloc[0] + tw // 2
            cy = y + maxloc[1] + th // 2
            best_score = float(maxv)
            best_center = (cx, cy)

    found = best_score >= threshold
    return found, best_center, best_score


def normalize_region(w, h, rx, ry, rw, rh):
    return (int(rx * w), int(ry * h), int(rw * w), int(rh * h))


def load_templates(templates_dir):
    templates = {}
    if templates_dir is None:
        return templates

    path = Path(templates_dir)
    if not path.exists():
        return templates

    for tpl_path in path.glob("*.png"):
        img = cv2.imread(str(tpl_path))
        if img is not None:
            templates[tpl_path.stem] = img
    return templates
