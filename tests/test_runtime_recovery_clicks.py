from megabonk_bot.runtime.recovery_clicks import make_window_click, try_click_template


def test_make_window_click_focuses_translates_coordinates_and_clicks():
    events = []

    class _Cap:
        def focus_if_needed(self, *, topmost, min_interval_s):
            events.append(("focus", topmost, min_interval_s))

        def get_bbox(self):
            events.append(("bbox",))
            return {"left": 11, "top": 22}

        def client_to_screen(self, x, y, *, bbox):
            events.append(("translate", x, y, bbox))
            return (x + 100, y + 200)

    class _Driver:
        def moveTo(self, x, y):
            events.append(("move", x, y))

        def click(self):
            events.append(("click",))

    delays = []
    click = make_window_click(
        _Cap(),
        input_driver=_Driver(),
        focus_interval_s=0.25,
        sleep_fn=delays.append,
    )

    click(10, 20, delay=0.05)

    assert events == [
        ("focus", False, 0.25),
        ("bbox",),
        ("translate", 10, 20, {"left": 11, "top": 22}),
        ("move", 110, 220),
        ("click",),
    ]
    assert delays == [0.05]


def test_try_click_template_uses_finder_and_click_handler():
    events = []

    def _finder(frame, tpl, region, threshold):
        events.append(("find", frame, tpl, region, threshold))
        return True, (30, 40), 0.91

    def _click(x, y, delay):
        events.append(("click", x, y, delay))

    ok = try_click_template(
        "frame",
        {"tpl_confirm": "confirm-template"},
        {"REG_DEAD_CONFIRM": (1, 2, 3, 4)},
        "tpl_confirm",
        "REG_DEAD_CONFIRM",
        0.6,
        click_fn=_click,
        finder=_finder,
    )

    assert ok is True
    assert events == [
        ("find", "frame", "confirm-template", (1, 2, 3, 4), 0.6),
        ("click", 30, 40, 0.0),
    ]


def test_try_click_template_returns_false_without_required_assets():
    ok = try_click_template(
        "frame",
        {},
        {"REG_MAIN_PLAY": (1, 2, 3, 4)},
        "tpl_play",
        "REG_MAIN_PLAY",
        0.65,
    )

    assert ok is False
