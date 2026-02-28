import torch
import yaml
import numpy as np
import argparse

from pathlib import Path
from PIL import Image
from src.detection.detection_model import faster_rcnn
from src.ocr.ocr_system import MangaTextExtractor
from src.translation.translator_system import MangaTranslator
from src.translation.renderer_system import MangaRenderer

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Input image file name")
    args = parser.parse_args()

    INPUT_DIR = Path("./data/input")
    OUTPUT_DIR = Path("./outputs")

    input_path = INPUT_DIR / args.filename
    output_path = OUTPUT_DIR / f"{input_path.stem}_translated{input_path.suffix}"

    INPUT_IMG_PATH = str(input_path)
    OUTPUT_IMG_PATH = str(output_path)
    
    CONFIG_PATH = "./configs/faster_rcnn_default.yaml"
    DETECTION_WEIGHTS_PATH = "./models/faster_rcnn_default_weights.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detection_model = load_trained_model(CONFIG_PATH, DETECTION_WEIGHTS_PATH, device)
    ocr_model = MangaTextExtractor()
    translator_module = MangaTranslator("ja", "en")
    renderer_module = MangaRenderer()

    # Detection
    confidence_threshold = 0.6

    img = Image.open(INPUT_IMG_PATH).convert("RGB")
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

    # OCR 
    ocr_results = ocr_model.extract_text(INPUT_IMG_PATH, text_boxes, frame_boxes)

    # Translation
    translated_data = translator_module.translate_with_context(ocr_results)

    # Render
    renderer_module.render_translated_image(INPUT_IMG_PATH, translated_data, OUTPUT_IMG_PATH)