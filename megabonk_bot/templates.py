from pathlib import Path

import cv2


def load_templates(templates_dir):
    templates = {}
    if templates_dir is None:
        return templates

    path = Path(templates_dir)
    if not path.exists():
        return templates

    for tpl_path in path.glob("*.png"):
        img = cv2.imread(str(tpl_path))
        if img is not None:
            templates[tpl_path.stem] = img
    return templates
