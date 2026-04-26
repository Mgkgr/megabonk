import json

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from megabonk_bot.asset_catalog import load_curated_catalogs
from megabonk_bot.memory_probe import ProbeResult
from megabonk_bot.recognition import analyze_scene
from megabonk_bot.regions import build_regions


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def test_analyze_scene_classifies_enemy_and_fuses_probe(tmp_path):
    asset_refs = tmp_path / "refs"
    preview = np.zeros((32, 32, 3), dtype=np.uint8)
    preview[:, :] = (0, 220, 0)
    cv2.rectangle(preview, (8, 8), (24, 24), (0, 255, 255), -1)
    _write_image(asset_refs / "enemies" / "bandit.png", preview)

    enemy_manifest = tmp_path / "enemy_catalog.json"
    enemy_manifest.write_text(
        json.dumps(
            [
                {
                    "entity_id": "bandit",
                    "kind": "enemy",
                    "display_name": "Bandit",
                    "aliases": ["bandit"],
                    "preview_relpath": "enemies/bandit.png",
                    "threat_tier": 2.0,
                    "metadata": {"family": "bandit", "variant": "default"},
                }
            ]
        ),
        encoding="utf-8",
    )
    world_manifest = tmp_path / "world_catalog.json"
    world_manifest.write_text("[]", encoding="utf-8")

    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_manifest,
        world_catalog_path=world_manifest,
    )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[50:82, 70:102] = preview
    regions = build_regions(320, 240)
    rx, ry, rw, rh = regions["REG_MINIMAP"]
    frame[ry : ry + rh, rx : rx + rw] = (20, 20, 20)
    frame[ry + 10 : ry + 14, rx + 20 : rx + 24] = (255, 255, 255)
    frame[ry + 25 : ry + 29, rx + 30 : rx + 34] = (0, 0, 255)

    analysis = analyze_scene(
        frame,
        catalogs=catalogs,
        regions=regions,
        probe_result=ProbeResult(
            status="ready",
            player_world_pos=(1.0, 2.0, 3.0),
            player_heading_deg=91.5,
            map_open=True,
            biome_or_scene_id="GeneratedMap",
            scene_id="GeneratedMap",
            active_room_or_node_id="room_1",
            current_objective="objective",
            ts=1.0,
        ),
        templates=None,
        enemy_hsv_lower=(35, 80, 80),
        enemy_hsv_upper=(95, 255, 255),
        enemy_min_area=10.0,
        enemy_classifier_mode="hybrid",
        minimap_enabled=True,
    )

    assert any(item.entity_id == "bandit" for item in analysis["enemy_classes"])
    assert analysis["player_pose"].world_pos == (1.0, 2.0, 3.0)
    assert analysis["player_pose"].heading_deg == 91.5
    assert analysis["map_state"].map_open is True
    assert analysis["map_state"].scene_id == "GeneratedMap"
    assert analysis["map_state"].objective == "objective"
    assert analysis["memory_probe_status"] == "ready"


def test_analyze_scene_detects_world_object_from_template(tmp_path):
    template = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.circle(template, (10, 10), 6, (255, 255, 255), -1)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[40:60, 50:70] = template

    world_manifest = tmp_path / "world_catalog.json"
    world_manifest.write_text(
        json.dumps(
            [
                {
                    "entity_id": "interactable_portal",
                    "kind": "poi",
                    "display_name": "Portal",
                    "aliases": ["portal"],
                    "template_names": ["tpl_portal"],
                    "poi_type": "exit",
                    "metadata": {"family": "portal", "variant": "normal"},
                }
            ]
        ),
        encoding="utf-8",
    )
    enemy_manifest = tmp_path / "enemy_catalog.json"
    enemy_manifest.write_text("[]", encoding="utf-8")
    catalogs = load_curated_catalogs(
        asset_refs_dir=tmp_path / "refs",
        enemy_catalog_path=enemy_manifest,
        world_catalog_path=world_manifest,
    )

    analysis = analyze_scene(
        frame,
        templates={"tpl_portal": template},
        catalogs=catalogs,
        regions=build_regions(120, 120),
        probe_result=ProbeResult(status="disabled", ts=1.0),
        enemy_hsv_lower=(35, 80, 80),
        enemy_hsv_upper=(95, 255, 255),
        enemy_min_area=50.0,
    )

    assert any(item.entity_id == "interactable_portal" for item in analysis["world_objects"])


def test_analyze_scene_uses_icon_atlas_for_minimap_pois(tmp_path):
    asset_refs = tmp_path / "refs"
    icon = np.zeros((12, 12, 3), dtype=np.uint8)
    pts = np.array([[6, 1], [10, 6], [6, 11], [1, 6]], dtype=np.int32)
    cv2.fillConvexPoly(icon, pts, (255, 0, 0))
    _write_image(asset_refs / "minimap_icons" / "portal.png", icon)

    enemy_manifest = tmp_path / "enemy_catalog.json"
    enemy_manifest.write_text("[]", encoding="utf-8")
    world_manifest = tmp_path / "world_catalog.json"
    world_manifest.write_text(
        json.dumps(
            [
                {
                    "entity_id": "interactable_portal",
                    "kind": "poi",
                    "display_name": "Portal",
                    "aliases": ["portal"],
                    "poi_type": "exit",
                    "metadata": {
                        "family": "portal",
                        "variant": "normal",
                        "icon_id": "portal",
                        "minimap_icon_relpath": "minimap_icons/portal.png",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_manifest,
        world_catalog_path=world_manifest,
    )

    frame = np.zeros((180, 280, 3), dtype=np.uint8)
    regions = build_regions(280, 180)
    rx, ry, rw, rh = regions["REG_MINIMAP"]
    frame[ry : ry + rh, rx : rx + rw] = (20, 20, 20)
    frame[ry + 8 : ry + 20, rx + 10 : rx + 22] = icon

    analysis = analyze_scene(
        frame,
        catalogs=catalogs,
        regions=regions,
        probe_result=ProbeResult(status="disabled", ts=1.0),
        enemy_hsv_lower=(35, 80, 80),
        enemy_hsv_upper=(95, 255, 255),
        enemy_min_area=40.0,
    )

    assert any(poi.icon_id == "portal" for poi in analysis["map_state"].pois)
    assert analysis["map_state"].source == "icon_atlas"


def test_analyze_scene_splits_projectiles_and_hazards(tmp_path):
    asset_refs = tmp_path / "refs"
    blood = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.circle(blood, (10, 10), 7, (0, 0, 255), -1)
    spike = np.zeros((28, 28, 3), dtype=np.uint8)
    cv2.rectangle(spike, (10, 2), (18, 26), (0, 140, 255), -1)
    _write_image(asset_refs / "enemies" / "bloodmagic.png", blood)
    _write_image(asset_refs / "enemies" / "ghost_spike.png", spike)

    enemy_manifest = tmp_path / "enemy_catalog.json"
    enemy_manifest.write_text("[]", encoding="utf-8")
    world_manifest = tmp_path / "world_catalog.json"
    world_manifest.write_text("[]", encoding="utf-8")
    projectile_manifest = tmp_path / "projectile_catalog.json"
    projectile_manifest.write_text(
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
                },
                {
                    "entity_id": "ghostboss_spike",
                    "kind": "hazard",
                    "display_name": "GhostBossSpike",
                    "aliases": ["ghostspike"],
                    "preview_relpath": "enemies/ghost_spike.png",
                    "threat_tier": 3.0,
                    "metadata": {"family": "ghostboss_spike", "hazard_kind": "spike"},
                }
            ]
        ),
        encoding="utf-8",
    )
    catalogs = load_curated_catalogs(
        asset_refs_dir=asset_refs,
        enemy_catalog_path=enemy_manifest,
        world_catalog_path=world_manifest,
        projectile_catalog_path=projectile_manifest,
    )

    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    frame[30:50, 30:50] = blood
    frame[76:104, 116:144] = spike

    analysis = analyze_scene(
        frame,
        catalogs=catalogs,
        regions=build_regions(220, 160),
        probe_result=ProbeResult(status="disabled", ts=1.0),
        enemy_hsv_lower=(35, 80, 80),
        enemy_hsv_upper=(95, 255, 255),
        enemy_min_area=40.0,
    )

    assert any(item.entity_id == "projectile_bloodmagic" for item in analysis["projectile_classes"])
    assert any(item.entity_id == "ghostboss_spike" for item in analysis["hazards"])


def test_finalboss_profile_filters_loot_world_objects(tmp_path):
    chest = np.zeros((18, 18, 3), dtype=np.uint8)
    cv2.rectangle(chest, (2, 2), (15, 15), (255, 255, 255), -1)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    frame[45:63, 45:63] = chest

    enemy_manifest = tmp_path / "enemy_catalog.json"
    enemy_manifest.write_text("[]", encoding="utf-8")
    world_manifest = tmp_path / "world_catalog.json"
    world_manifest.write_text(
        json.dumps(
            [
                {
                    "entity_id": "chest_free_crypt",
                    "kind": "world_object",
                    "display_name": "ChestFreeCrypt",
                    "aliases": ["chestfreecrypt"],
                    "template_names": ["tpl_chest"],
                    "poi_type": "loot",
                    "metadata": {"family": "chest", "variant": "free_crypt"},
                }
            ]
        ),
        encoding="utf-8",
    )
    catalogs = load_curated_catalogs(
        asset_refs_dir=tmp_path / "refs",
        enemy_catalog_path=enemy_manifest,
        world_catalog_path=world_manifest,
    )

    analysis = analyze_scene(
        frame,
        templates={"tpl_chest": chest},
        catalogs=catalogs,
        regions=build_regions(120, 120),
        probe_result=ProbeResult(status="ready", scene_id="FinalBossMap", ts=1.0),
        enemy_hsv_lower=(35, 80, 80),
        enemy_hsv_upper=(95, 255, 255),
        enemy_min_area=40.0,
    )

    assert analysis["detector_profile"].scene_id == "FinalBossMap"
    assert analysis["world_objects"] == []
