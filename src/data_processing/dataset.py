import json
import torch
import numpy as np
import albumentations as A

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class Manga109Dataset(Dataset):
    def __init__(self, json_file, img_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        with open(json_file, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.imgs = {img["id"]: img for img in self.coco_data["images"]}
        self.annos = {}
        for anno in self.coco_data["annotations"]:
            img_id = anno["image_id"]
            if img_id not in self.annos:
                self.annos[img_id] = []
            self.annos[img_id].append(anno)

        self.img_ids = list(self.imgs.keys())

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.imgs[img_id]
        img_path = self.img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        annos = self.annos.get(img_id, [])
        boxes = []
        labels = []
        iscrowd = []

        for anno in annos:
            x, y, w, h = anno["bbox"]
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h

            if w <= 0 or h <= 0:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(anno["category_id"]))
            iscrowd.append(int(anno.get("iscrowd", 0)))

        if self.transforms:
            transformed = self.transforms(image=img_np, bboxes=boxes, labels=labels)
            img_tensor = transformed["image"]
            img_tensor = img_tensor.to(torch.float32) / 255.0
            boxes = transformed.get("bboxes", [])
            labels = transformed.get("labels", [])
        else:
            img_tensor = torch.tensor((img_np / 255.0), dtype=torch.float32).permute(2,0,1)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            labels = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)      
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            areas = torch.zeros((0, ), dtype=torch.float32)
            iscrowd = torch.zeros((0, ), dtype=torch.int64)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([int(img_id)], dtype=torch.int64)
        }

        return img_tensor, target