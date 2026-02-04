def build_regions(width, height):
    return {
        "REG_MAIN_PLAY": (
            int(0.40 * width),
            int(0.35 * height),
            int(0.25 * width),
            int(0.20 * height),
        ),
        "REG_CHAR_SELECT": (
            int(0.02 * width),
            int(0.02 * height),
            int(0.40 * width),
            int(0.12 * height),
        ),
        "REG_CHAR_GRID": (
            int(0.02 * width),
            int(0.12 * height),
            int(0.45 * width),
            int(0.75 * height),
        ),
        "REG_CHAR_CONFIRM": (
            int(0.70 * width),
            int(0.70 * height),
            int(0.25 * width),
            int(0.25 * height),
        ),
        "REG_DEAD_CONFIRM": (
            int(0.30 * width),
            int(0.55 * height),
            int(0.40 * width),
            int(0.30 * height),
        ),
        "REG_UNLOCKS": (
            int(0.02 * width),
            int(0.02 * height),
            int(0.40 * width),
            int(0.10 * height),
        ),
        "REG_CHEST": (
            int(0.05 * width),
            int(0.12 * height),
            int(0.90 * width),
            int(0.80 * height),
        ),
        "REG_HUD": (
            int(0.70 * width),
            int(0.00 * height),
            int(0.30 * width),
            int(0.12 * height),
        ),
        "REG_HUD_HP": (
            int(0.04 * width),
            int(0.02 * height),
            int(0.12 * width),
            int(0.05 * height),
        ),
        "REG_HUD_GOLD": (
            int(0.85 * width),
            int(0.02 * height),
            int(0.12 * width),
            int(0.05 * height),
        ),
        "REG_HUD_TIME": (
            int(0.41 * width),
            int(0.01 * height),
            int(0.18 * width),
            int(0.06 * height),
        ),
        "REG_MINIMAP": (
            int(0.80 * width),
            int(0.05 * height),
            int(0.20 * width),
            int(0.30 * height),
        ),
        "REG_DEAD": (
            int(0.30 * width),
            int(0.20 * height),
            int(0.40 * width),
            int(0.20 * height),
        ),
    }
