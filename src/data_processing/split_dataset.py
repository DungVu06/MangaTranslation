import random
import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOKS_TXT = ROOT / "data/raw/books.txt"
INPUT_JSON = ROOT / "data/processed/manga109_coco.json"
OUTPUT_DIR = ROOT / "data/processed"

random.seed(42)

with open(BOOKS_TXT, "r", encoding="utf-8") as f:
    books = [line.strip() for line in f.readlines() if line.strip()]

random.shuffle(books)

train_books = books[:62]
val_books = books[62:75]
test_books = books[75:]

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

def create_split_json(target_books, output_filename):
    split_data = {
        "info": coco_data.get("info", {}),
        "categories": coco_data["categories"],
        "images": [],
        "annotations": []
    }

    valid_img_ids = set()

    for img in coco_data["images"]:
        book_name = img["file_name"].split("/")[0]
        if book_name in target_books:
            split_data["images"].append(img)
            valid_img_ids.add(img["id"])

    for anno in coco_data["annotations"]:
        if anno["image_id"] in valid_img_ids:
            split_data["annotations"].append(anno)
    
    output_path = OUTPUT_DIR / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)

create_split_json(train_books, "train_coco.json")
create_split_json(val_books, "val_coco.json")
create_split_json(test_books, "test_coco.json")