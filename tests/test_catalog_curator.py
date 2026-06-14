import json

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

from megabonk_bot.catalog_curator import curate_samples, parse_dropped_paths


def test_parse_dropped_paths_supports_windows_drag_and_drop_text():
    parsed = parse_dropped_paths('"C:\\tmp\\a one.png" "D:\\shots\\b-two.jpg"')
    assert parsed == ["C:\\tmp\\a one.png", "D:\\shots\\b-two.jpg"]


def test_curate_samples_writes_enemy_yaml_and_copies_images(tmp_path):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    source_image = source_dir / "orc shot.png"
    sample = np.zeros((20, 24, 3), dtype=np.uint8)
    sample[:, :] = (0, 220, 0)
    cv2.imwrite(str(source_image), sample)

    manifest_path = tmp_path / "enemy_catalog.yaml"
    asset_refs_dir = tmp_path / "refs"
    result = curate_samples(
        category="enemies",
        entity_id="orc",
        display_name="Orc",
        input_paths=[source_image],
        manifest_path=manifest_path,
        asset_refs_dir=asset_refs_dir,
        aliases=("orc", "brute"),
    )

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert [entry["entity_id"] for entry in payload] == ["orc"]
    assert payload[0]["kind"] == "enemy"
    assert payload[0]["preview_relpath"].startswith("imported/enemies/orc/")
    assert payload[0]["aliases"] == ["orc", "brute"]
    assert (asset_refs_dir / payload[0]["preview_relpath"]).exists()
    assert result.copied_files[0].exists()


def test_curate_samples_updates_existing_world_yaml_entry(tmp_path):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    first_image = source_dir / "chest_1.png"
    second_image = source_dir / "chest_2.png"
    sample = np.zeros((18, 18, 3), dtype=np.uint8)
    cv2.rectangle(sample, (3, 3), (14, 14), (255, 255, 255), -1)
    cv2.imwrite(str(first_image), sample)
    cv2.imwrite(str(second_image), sample)

    manifest_path = tmp_path / "world_catalog.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            [
                {
                    "entity_id": "chest",
                    "kind": "poi",
                    "display_name": "Chest",
                    "aliases": ["chest"],
                    "preview_relpath": "imported/props/chest/old.png",
                    "poi_type": "loot",
                    "metadata": {"family": "chest", "variant": "default"},
                }
            ],
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    curate_samples(
        category="props",
        entity_id="chest",
        display_name="Chest",
        input_paths=[first_image, second_image],
        manifest_path=manifest_path,
        asset_refs_dir=tmp_path / "refs",
        aliases=("chest", "loot"),
        family="chest",
        poi_type="loot",
    )

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["entity_id"] == "chest"
    assert payload[0]["aliases"] == ["chest", "loot"]
    assert payload[0]["poi_type"] == "loot"
    assert payload[0]["metadata"]["family"] == "chest"
