import cv2
import numpy as np
from pathlib import Path


def match_template(gray_img, gray_tpl):
    res = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCOEFF_NORMED)
    minv, maxv, minp, maxp = cv2.minMaxLoc(res)
    return maxv, maxp  # score, top-left


def find_in_region(frame_bgr, tpl_bgr, region, threshold=0.80):
    """
    region: (x,y,w,h) в координатах кадра
    Возвращает (found, (cx,cy), score)
    """
    x, y, w, h = region
    roi = frame_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return False, (0, 0), 0.0

    g_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g_tpl = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

    score, (tx, ty) = match_template(g_roi, g_tpl)
    if score < threshold:
        return False, (0, 0), score

    th, tw = g_tpl.shape[:2]
    cx = x + tx + tw // 2
    cy = y + ty + th // 2
    return True, (cx, cy), score


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
