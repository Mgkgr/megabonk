from megabonk_bot.env.hud_worker import HudDumpPolicyState, should_dump_hud_debug


def test_hud_dump_policy_off_never_dumps():
    state = HudDumpPolicyState()
    assert (
        should_dump_hud_debug(
            state=state,
            policy="off",
            now=1.0,
            fail_reason=None,
            min_interval_s=15.0,
            startup=True,
        )
        is False
    )
    assert (
        should_dump_hud_debug(
            state=state,
            policy="off",
            now=2.0,
            fail_reason="tesseract_missing",
            min_interval_s=15.0,
        )
        is False
    )


def test_hud_dump_policy_on_fail_change():
    state = HudDumpPolicyState()
    assert (
        should_dump_hud_debug(
            state=state,
            policy="on_fail_change",
            now=1.0,
            fail_reason=None,
            min_interval_s=15.0,
            startup=True,
        )
        is True
    )
    # Первый fail после нормального состояния.
    assert (
        should_dump_hud_debug(
            state=state,
            policy="on_fail_change",
            now=2.0,
            fail_reason="ocr_empty",
            min_interval_s=15.0,
        )
        is True
    )
    # Та же причина и маленький интервал -> дампа нет.
    assert (
        should_dump_hud_debug(
            state=state,
            policy="on_fail_change",
            now=3.0,
            fail_reason="ocr_empty",
            min_interval_s=15.0,
        )
        is False
    )
    # Смена причины -> новый дамп.
    assert (
        should_dump_hud_debug(
            state=state,
            policy="on_fail_change",
            now=4.0,
            fail_reason="tesseract_missing",
            min_interval_s=15.0,
        )
        is True
    )
