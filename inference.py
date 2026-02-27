import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from pathlib import Path
from src.detection.detection_model import faster_rcnn
from src.ocr.ocr_system import MangaTextExtractor
from src.translation.translator_system import MangaTranslator

plt.rcParams['font.family'] = 'MS Gothic'

def load_trained_model(config_path, weights, device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = faster_rcnn(
        num_classes=config["model"]["num_classes"],
        anchor_sizes=config["model"]["anchor_sizes"],
        anchor_ratios=config["model"]["anchor_ratios"] 
    )

    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    return model

def inference(img_path, detection_model, ocr_model, translator, device, confidence_threshold=0.6):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.inference_mode():
        predictions = detection_model(img_tensor)[0]

    keep_idx = predictions["scores"] >= confidence_threshold
    boxes = predictions["boxes"][keep_idx].cpu().numpy()
    labels = predictions["labels"][keep_idx].cpu().numpy()
    
    text_boxes = []
    frame_boxes = []

    for box, label in zip(boxes, labels):
        if label == 1:
            frame_boxes.append(box)
        else:
            text_boxes.append(box)
    
    ocr_results = ocr_model.extract_text(img_path, text_boxes, frame_boxes)
    traslation_results = translator.translate_with_context(ocr_results)

    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.imshow(img_np)

    for box in frame_boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin,
            linewidth=2, edgecolor="blue", facecolor="none"
        )
        ax.add_patch(rect)

    for result in traslation_results:
        box_id = result["box_id"]
        xmin, ymin, xmax, ymax = result["coordinates"]
        jp_text = result["japanese_text"]
        en_text = result["english_text"]
        print(f"{box_id}:\n {jp_text} -> {en_text}")

        rect = patches.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            xmin, ymin-5, f"{box_id}", color="white", fontsize=12,
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', pad=3)
        )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    CONFIG_PATH = "./configs/faster_rcnn_default.yaml"
    WEIGHTS_PATH = "./models/faster_rcnn_default_weights.pt"
    IMG_INFERENCE_PATH = "./data/inference_data/doraemon_1.jpg"
    device = "cpu"

    detection_model = load_trained_model(CONFIG_PATH, WEIGHTS_PATH, device)
    ocr_model = MangaTextExtractor()
    translator = MangaTranslator("ja", "en")

    inference(IMG_INFERENCE_PATH, detection_model, ocr_model, translator, device)