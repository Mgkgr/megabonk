import logging
import sys
import types

import pytest

from megabonk_bot.asset_catalog import OcrLexicon
from megabonk_bot.ui_ocr import lexicon_whitelist, normalize_with_lexicon, read_objective_ui


def test_normalize_with_lexicon_prefers_curated_mapping():
    lexicon = OcrLexicon(
        tokens=("challenge shrine", "objective arrow"),
        phrases=("ChallengeModifierCryptEscape",),
        normalize={
            "objective arrow": "objective",
            "challengemodifiercryptescape": "challenge_escape_first_crypt",
        },
    )

    assert normalize_with_lexicon("Objective Arrow", lexicon) == "objective"
    assert normalize_with_lexicon("ChallengeModifierCryptEscape", lexicon) == "challenge_escape_first_crypt"
    assert "O" in lexicon_whitelist(lexicon)


def test_read_objective_ui_logs_ocr_failure(monkeypatch, caplog):
    np = pytest.importorskip("numpy")

    def _boom(*args, **kwargs):
        raise RuntimeError("ocr backend exploded")

    monkeypatch.setitem(sys.modules, "megabonk_bot.hud", types.SimpleNamespace(_best_ocr=_boom))
    caplog.set_level(logging.WARNING)

    detection = read_objective_ui(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert detection.source == "none"
    assert any("Objective UI OCR failed" in record.message for record in caplog.records)
