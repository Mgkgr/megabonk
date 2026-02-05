from megabonk_bot.regions import build_regions


def test_build_regions_has_expected_keys():
    regions = build_regions(1920, 1080)
    for key in ["REG_MAIN_PLAY", "REG_CHAR_GRID", "REG_CHEST"]:
        assert key in regions
    for x, y, w, h in regions.values():
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0


def test_hud_time_region_matches_fixed_top_left_box():
    regions = build_regions(1920, 1080)
    assert regions["REG_HUD_TIME"] == (28, 61, 127, 41)


def test_hud_time_region_clamps_to_small_frame():
    regions = build_regions(100, 80)
    x, y, w, h = regions["REG_HUD_TIME"]
    assert (x, y) == (28, 61)
    assert w > 0 and h > 0
    assert x + w <= 100
    assert y + h <= 80
