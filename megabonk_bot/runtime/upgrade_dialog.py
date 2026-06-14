from __future__ import annotations


def is_upgrade_dialog(frame_bgr, templates, regions, threshold: float = 0.62) -> bool:
    from megabonk_bot.vision import find_in_region

    if frame_bgr is None or not templates or not regions:
        return False
    region = regions.get("REG_CHEST")
    if not region:
        return False
    for name in (
        "tpl_katana",
        "tpl_dexec",
        "tpl_foliant_bottom1",
        "tpl_foliant_bottom2",
        "tpl_foliant_bottom3",
        "tpl_blood_tome",
    ):
        tpl = templates.get(name)
        if tpl is None:
            continue
        found, _, _ = find_in_region(frame_bgr, tpl, region, threshold=threshold)
        if found:
            return True
    return False
