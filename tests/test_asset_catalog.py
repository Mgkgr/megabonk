import json

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from megabonk_bot.asset_catalog import load_curated_catalogs


def test_load_curated_catalogs_missing_paths_is_safe(tmp_path):
    catalogs = load_curated_catalogs(
        asset_refs_dir=tmp_path / "missing_refs",
        enemy_catalog_path=tmp_path / "missing_enemy.json",
        world_catalog_path=tmp_path / "missing_world.json",
        projectile_catalog_path=tmp_path / "missing_projectiles.json",
        ocr_lexicon_path=tmp_path / "missing_lexicon.json",
    )
    assert catalogs.enemies == ()
    assert catalogs.world == ()
    assert catalogs.projectiles == ()
    assert catalogs.ocr_lexicon.tokens == ()
    assert catalogs.minimap_icon_entries() == ()


def test_load_curated_catalogs_loads_previews_icons_silhouettes_and_lexicon(tmp_path):
    asset_refs = tmp_path / "refs"
    enemies_dir = asset_refs / "enemies"
    icons_dir = asset_refs / "minimap_icons"
    silhouettes_dir = asset_refs / "silhouettes"
    for directory in (enemies_dir, icons_dir, silhouettes_dir):
        directory.mkdir(parents=True)

    preview = np.zeros((16, 16, 3), dtype=np.uint8)
    preview[:, :] = (0, 255, 0)
    icon = np.zeros((10, 10, 3), dtype=np.uint8)
    icon[:, :] = (255, 0, 0)
    silhouette = np.zeros((12, 12, 3), dtype=np.uint8)
    cv2.rectangle(silhouette, (2, 2), (9, 9), (255, 255, 255), -1)
    projectile = np.zeros((14, 14, 3), dtype=np.uint8)
    cv2.circle(projectile, (7, 7), 4, (0, 0, 255), -1)

    cv2.imwrite(str(enemies_dir / "bandit.png"), preview)
    cv2.imwrite(str(icons_dir / "boss.png"), icon)
    cv2.imwrite(str(silhouettes_dir / "Bandit.png"), silhouette)
    cv2.imwrite(str(enemies_dir / "bloodmagic.png"), projectile)

    enemy_manifest_path = tmp_path / "enemy_catalog.json"
    enemy_manifest_path.write_text(
        json.dumps(
            [
                {
                    "entity_id": "bandit",
                    "kind": "enemy",
                    "display_name": "Bandit",
                    "aliases": ["bandit"],
                    "preview_relpath": "enemies/bandit.png",
                    "threat_tier": 2.0,
                    "metadata": {
                        "family": "bandit",
                        "variant": "default",
                        "minimap_icon_relpath": "minimap_icons/boss.png",
                        "silhouette_relpath": "silhouettes/Bandit.png",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    world_manifest_path = tmp_path / "world_catalog.json"
    world_manifest_path.write_text("[]", encoding="utf-8")
    projectile_manifest_path = tmp_path / "projectile_catalog.json"
    projectile_manifest_path.write_text(
        json.dumps(
            [
                {
                    "entity_id": "projectile_bloodmagic",
                    "kind": "projectile",
                    "display_name": "ProjectileBloodMagic",
                    "aliases": ["bloodmagic"],
                    "preview_relpath": "enemies/bloodmagic.png",
                    "threat_tier": 2.6,
                    "metadata": {"family": "projectile_bloodmagic", "damage_type": "blood"},
                }
            ]
        ),
        encoding="utf-8",
    )
    lexicon_path = tmp_path / "ocr_lexicon.json"
    lexicon_path.write_text(
        json.dumps(
            {
                "tokens": ["challenge shrine", "objective arrow"],
                "phrases": ["ChallengeModifierCryptEscape"],
                "normalize": {"objective arrow": "objective"},
            }
        ),
        encoding="utf-8",
    )

    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_manifest_path,
        world_catalog_path=world_manifest_path,
        projectile_catalog_path=projectile_manifest_path,
        ocr_lexicon_path=lexicon_path,
    )

    assert len(catalogs.enemies) == 1
    assert catalogs.enemies[0].preview_bgr is not None
    assert catalogs.enemies[0].preview_size == (16, 16)
    assert catalogs.enemies[0].minimap_icon_bgr is not None
    assert catalogs.enemies[0].silhouette_bgr is not None
    assert catalogs.enemies[0].family == "bandit"
    assert len(catalogs.projectiles) == 1
    assert catalogs.projectiles[0].damage_type == "blood"
    assert catalogs.ocr_lexicon.normalize["objective arrow"] == "objective"
    assert len(catalogs.minimap_icon_entries()) == 1
