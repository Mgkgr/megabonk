from pathlib import Path


def test_no_merge_conflict_markers_in_python_files():
    root = Path(__file__).resolve().parents[1]
    markers = (("<" * 7) + " ", "=" * 7, (">" * 7) + " ")
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}
    bad = []
    for path in root.rglob("*.py"):
        if any(part in ignored_parts for part in path.parts):
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            stripped = line.lstrip()
            if any(stripped.startswith(marker) for marker in markers):
                bad.append(str(path.relative_to(root)))
                break
    assert not bad, f"Conflict markers found in: {bad}"
