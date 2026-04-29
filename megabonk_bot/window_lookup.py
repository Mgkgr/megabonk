from __future__ import annotations

from collections.abc import Iterable


def select_window_title_match(
    title_substr: str,
    candidates: Iterable[tuple[int, str]],
) -> int | None:
    target = str(title_substr or "").strip().lower()
    substring_matches: list[tuple[int, int]] = []

    for hwnd, title in candidates:
        title_lower = str(title or "").lower()
        if target == title_lower:
            return int(hwnd)
        if target in title_lower:
            substring_matches.append((len(title_lower), int(hwnd)))

    if not substring_matches:
        return None
    substring_matches.sort(key=lambda item: item[0])
    return substring_matches[0][1]
