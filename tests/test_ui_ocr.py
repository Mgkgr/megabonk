from megabonk_bot.asset_catalog import OcrLexicon
from megabonk_bot.ui_ocr import lexicon_whitelist, normalize_with_lexicon


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
