import shutil

from pathlib import Path

def sanitize_data(root_path):
    root_path = Path(root_path)

    all_paths = list(root_path.rglob("*"))
    all_paths.reverse()

    for path in all_paths:
        if "'" in path.name:
            new_name = path.name.replace("'", "_")
            new_path = path.with_name(new_name)
            path = path.rename(new_path)

def sanitize_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content.replace("That'sIzumiko", "That_sIzumiko").replace("UchiNoNyan'sDiary", "UchiNoNyan_sDiary")
    
    with open(json_file, "w", encoding="utf-8") as f:
        f.write(new_content)

current_path = Path(__file__).resolve()
ROOT = current_path.parent.parent.parent

sanitize_data(ROOT / "data/raw/images")
sanitize_json(ROOT / "data/processed/train_coco.json")
sanitize_json(ROOT / "data/processed/val_coco.json")
sanitize_json(ROOT / "data/processed/test_coco.json")