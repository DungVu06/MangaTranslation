import manga109api
import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_ROOT = ROOT / "data/raw"
OUTPUT_JSON_PATH = ROOT / "data/processed/manga109_coco.json"

CATEGORY_MAP = {"frame": 1, "text": 2}

coco_format = {
    "info": {"description": "Manga109 Dataset"},
    "categories": [
        {"id": 1, "name": "frame"},
        {"id": 2, "name": "text"}
    ],
    "images": [],
    "annotations": []
}
parser = manga109api.Parser(root_dir=RAW_DATA_ROOT)

img_id = 0
anno_id = 0

for book in parser.books:
    annotations = parser.get_annotation(book=book)

    for page in annotations["page"]:
        img_width = float(page["@width"])
        img_height = float(page["@height"])

        page_index = int(page["@index"])
        file_name = f"{book}/{page_index:03d}.jpg"

        coco_format["images"].append({
            "id": img_id,
            "width": img_width,
            "height": img_height,
            "file_name": file_name
        })

        for rois, cat_name in [("frame", "frame"), ("text", "text")]:
            if rois in page:
                for roi in page[rois]:
                    xmin = float(roi["@xmin"])
                    xmax = float(roi["@xmax"])
                    ymin = float(roi["@ymin"])
                    ymax = float(roi["@ymax"])

                    width = xmax - xmin
                    height = ymax - ymin
                    area = width * height

                    coco_format["annotations"].append({
                        "id": anno_id,
                        "image_id": img_id,
                        "category_id": CATEGORY_MAP[cat_name],
                        "bbox": [xmin, ymin, width, height],
                        "area": area,
                        "iscrowd": 0
                    })
                    anno_id += 1
        img_id += 1       

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(obj=coco_format, fp=f, ensure_ascii=False, indent=2)