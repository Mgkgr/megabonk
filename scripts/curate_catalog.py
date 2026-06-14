from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megabonk_bot.catalog_curator import run_interactive_curator


if __name__ == "__main__":
    run_interactive_curator(project_root=ROOT)
