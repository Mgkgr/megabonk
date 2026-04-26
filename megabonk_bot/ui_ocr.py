from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from megabonk_bot.asset_catalog import OcrLexicon


@dataclass(frozen=True)
class UiTextDetection:
    text: str | None = None
    normalized: str | None = None
    confidence: float = 0.0
    region: tuple[int, int, int, int] | None = None
    source: str = "none"


def _normalize_free_text(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "")).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def lexicon_whitelist(lexicon: OcrLexicon | None) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
    if lexicon is not None:
        for item in lexicon.all_terms():
            allowed.update(ch for ch in str(item) if ch.isalnum() or ch in {"_", "-"})
    return "".join(sorted(allowed))


def normalize_with_lexicon(text: str | None, lexicon: OcrLexicon | None) -> str | None:
    normalized = _normalize_free_text(text or "")
    if not normalized:
        return None
    if lexicon is None:
        return normalized

    norm_map = { _normalize_free_text(key): str(value) for key, value in dict(lexicon.normalize).items() if key and value }
    if normalized in norm_map:
        return norm_map[normalized]

    phrase_hits = []
    for phrase in lexicon.phrases:
        phrase_norm = _normalize_free_text(phrase)
        if phrase_norm and phrase_norm in normalized:
            phrase_hits.append((len(phrase_norm), phrase_norm))
    if phrase_hits:
        _, best_phrase = max(phrase_hits, key=lambda item: item[0])
        return norm_map.get(best_phrase, best_phrase.replace(" ", "_"))

    tokens = set(normalized.split())
    best_score = 0.0
    best_match = None
    for candidate in lexicon.tokens:
        candidate_norm = _normalize_free_text(candidate)
        if not candidate_norm:
            continue
        candidate_tokens = set(candidate_norm.split())
        overlap = len(tokens & candidate_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, len(candidate_tokens))
        if score > best_score:
            best_score = score
            best_match = candidate_norm
    if best_match is not None and best_score >= 0.6:
        return norm_map.get(best_match, best_match.replace(" ", "_"))
    return normalized.replace(" ", "_")


def objective_region(frame_bgr) -> tuple[int, int, int, int] | None:
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    h, w = frame_bgr.shape[:2]
    x = int(w * 0.22)
    y = int(h * 0.06)
    rw = int(w * 0.56)
    rh = int(h * 0.12)
    return (x, y, rw, rh)


def read_objective_ui(frame_bgr, *, lexicon: OcrLexicon | None = None) -> UiTextDetection:
    region = objective_region(frame_bgr)
    if region is None:
        return UiTextDetection()
    x, y, w, h = region
    roi = frame_bgr[y : y + h, x : x + w]
    if getattr(roi, "size", 0) == 0:
        return UiTextDetection(region=region)
    try:
        from megabonk_bot.hud import _best_ocr
    except Exception:
        return UiTextDetection(region=region)

    whitelist = lexicon_whitelist(lexicon)
    try:
        text, conf = _best_ocr(roi, whitelist=whitelist, psm=6, target_conf=45.0)
    except Exception:
        return UiTextDetection(region=region)
    if text is None or conf is None:
        return UiTextDetection(region=region)
    normalized = normalize_with_lexicon(text, lexicon)
    return UiTextDetection(
        text=str(text),
        normalized=normalized,
        confidence=float(conf),
        region=region,
        source="ui_ocr",
    )
