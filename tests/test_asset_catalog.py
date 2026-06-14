import json
from pathlib import Path

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


def test_load_curated_catalogs_attaches_dbg_hud_enemy_samples(tmp_path):
    asset_refs = tmp_path / "refs"
    enemies_dir = asset_refs / "enemies"
    enemies_dir.mkdir(parents=True)
    (tmp_path / "dbg_hud").mkdir(parents=True)

    preview = np.zeros((16, 16, 3), dtype=np.uint8)
    preview[:, :] = (255, 0, 255)
    extra = np.zeros((18, 18, 3), dtype=np.uint8)
    extra[:, :] = (0, 220, 0)

    cv2.imwrite(str(enemies_dir / "orc_main.png"), preview)
    cv2.imwrite(str(tmp_path / "dbg_hud" / "3orc.png"), extra)

    enemy_manifest_path = tmp_path / "enemy_catalog.json"
    enemy_manifest_path.write_text(
        json.dumps(
            [
                {
                    "entity_id": "orc",
                    "kind": "enemy",
                    "display_name": "Orc",
                    "aliases": ["orc", "fbx_orc_Color"],
                    "preview_relpath": "enemies/orc_main.png",
                    "threat_tier": 3.0,
                    "metadata": {"family": "orc", "variant": "default"},
                }
            ]
        ),
        encoding="utf-8",
    )
    world_manifest_path = tmp_path / "world_catalog.json"
    world_manifest_path.write_text("[]", encoding="utf-8")

    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_manifest_path,
        world_catalog_path=world_manifest_path,
    )

    assert len(catalogs.enemies) == 1
    assert catalogs.enemies[0].preview_size == (16, 16)
    assert catalogs.enemies[0].extra_preview_sizes == ((18, 18),)
    assert catalogs.enemies[0].extra_preview_paths[0].name == "3orc.png"
    assert len(catalogs.enemies[0].preview_samples) == 2


def test_load_curated_catalogs_attaches_dbg_hud_enemy_samples_from_subdir(tmp_path):
    asset_refs = tmp_path / "refs"
    enemies_dir = asset_refs / "enemies"
    enemies_dir.mkdir(parents=True)
    (tmp_path / "dbg_hud" / "orc").mkdir(parents=True)

    preview = np.zeros((16, 16, 3), dtype=np.uint8)
    preview[:, :] = (255, 0, 255)
    extra = np.zeros((18, 18, 3), dtype=np.uint8)
    extra[:, :] = (0, 220, 0)

    cv2.imwrite(str(enemies_dir / "orc_main.png"), preview)
    cv2.imwrite(str(tmp_path / "dbg_hud" / "orc" / "4orc.png"), extra)

    enemy_manifest_path = tmp_path / "enemy_catalog.json"
    enemy_manifest_path.write_text(
        json.dumps(
            [
                {
                    "entity_id": "orc",
                    "kind": "enemy",
                    "display_name": "Orc",
                    "aliases": ["orc", "fbx_orc_Color"],
                    "preview_relpath": "enemies/orc_main.png",
                    "threat_tier": 3.0,
                    "metadata": {"family": "orc", "variant": "default"},
                }
            ]
        ),
        encoding="utf-8",
    )
    world_manifest_path = tmp_path / "world_catalog.json"
    world_manifest_path.write_text("[]", encoding="utf-8")

    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_manifest_path,
        world_catalog_path=world_manifest_path,
    )

    assert len(catalogs.enemies) == 1
    assert [path.name for path in catalogs.enemies[0].extra_preview_paths] == ["4orc.png"]
    assert len(getattr(catalogs.enemies[0], "preview_descriptors", ())) == 2


def test_dbg_hud_orc_samples_do_not_attach_to_color_aliases(tmp_path):
    (tmp_path / "dbg_hud").mkdir(parents=True)
    sample = np.zeros((18, 18, 3), dtype=np.uint8)
    sample[:, :] = (0, 220, 0)
    cv2.imwrite(str(tmp_path / "dbg_hud" / "orc.png"), sample)

    enemy_manifest_path = tmp_path / "enemy_catalog.json"
    enemy_manifest_path.write_text(
        json.dumps(
            [
                {
                    "entity_id": "orc",
                    "kind": "enemy",
                    "display_name": "Orc",
                    "aliases": ["orc", "fbx_orc_Color"],
                    "preview_relpath": "missing/orc.png",
                    "metadata": {"family": "orc"},
                },
                {
                    "entity_id": "shadyguy",
                    "kind": "enemy",
                    "display_name": "ShadyGuy",
                    "aliases": ["shady", "shadyguy", "fbx_ShadyGuy_ColorCommon"],
                    "preview_relpath": "missing/shady.png",
                    "metadata": {"family": "shadyguy"},
                },
            ]
        ),
        encoding="utf-8",
    )
    world_manifest_path = tmp_path / "world_catalog.json"
    world_manifest_path.write_text("[]", encoding="utf-8")

    catalogs = load_curated_catalogs(
        asset_refs_dir=tmp_path / "missing_refs",
        enemy_catalog_path=enemy_manifest_path,
        world_catalog_path=world_manifest_path,
    )
    by_id = {entry.entity_id: entry for entry in catalogs.enemies}

    assert [path.name for path in by_id["orc"].extra_preview_paths] == ["orc.png"]
    assert by_id["shadyguy"].extra_preview_paths == ()


def test_load_curated_catalogs_reads_yaml_manifest(tmp_path):
    pytest.importorskip("yaml")
    asset_refs = tmp_path / "refs"
    enemies_dir = asset_refs / "imported" / "enemies" / "orc"
    enemies_dir.mkdir(parents=True)

    preview = np.zeros((16, 16, 3), dtype=np.uint8)
    preview[:, :] = (0, 220, 0)
    cv2.imwrite(str(enemies_dir / "sample.png"), preview)

    enemy_manifest_path = tmp_path / "enemy_catalog.yaml"
    enemy_manifest_path.write_text(
        "\n".join(
            [
                "- entity_id: orc",
                "  kind: enemy",
                "  display_name: Orc",
                "  aliases: [orc, brute]",
                "  preview_relpath: imported/enemies/orc/sample.png",
                "  threat_tier: 3.0",
                "  metadata:",
                "    family: orc",
                "    variant: default",
            ]
        ),
        encoding="utf-8",
    )
    world_manifest_path = tmp_path / "world_catalog.yaml"
    world_manifest_path.write_text("[]\n", encoding="utf-8")

    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_manifest_path,
        world_catalog_path=world_manifest_path,
    )

    assert len(catalogs.enemies) == 1
    assert catalogs.enemies[0].entity_id == "orc"
    assert catalogs.enemies[0].aliases == ("orc", "brute")
    assert catalogs.enemies[0].preview_size == (16, 16)


def test_repo_survival_catalog_manifests_have_preview_samples_for_all_entries():
    pytest.importorskip("yaml")
    repo_root = Path(__file__).resolve().parents[1]

    catalogs = load_curated_catalogs(
        asset_refs_dir=repo_root / "art_refs" / "megabonk_unity_extracts",
        enemy_catalog_path=repo_root / "config" / "enemy_catalog.yaml",
        world_catalog_path=repo_root / "config" / "world_catalog.yaml",
        projectile_catalog_path=repo_root / "config" / "projectile_catalog.json",
    )

    assert catalogs.enemies
    assert catalogs.world
    assert all(entry.preview_samples for entry in catalogs.enemies)
    assert all(entry.preview_samples for entry in catalogs.world)
