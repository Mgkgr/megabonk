from megabonk_bot.regions import build_regions


def test_build_regions_has_expected_keys():
    regions = build_regions(1920, 1080)
    for key in ["REG_MAIN_PLAY", "REG_CHAR_GRID", "REG_CHEST"]:
        assert key in regions
    for x, y, w, h in regions.values():
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0
