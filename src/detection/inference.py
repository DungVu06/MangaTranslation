import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from pathlib import Path
from src.detection.detection_model import faster_rcnn

def load_trained_model_cpu(config_path, weights, device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(config)
    model = faster_rcnn(
        num_classes=config["model"]["num_classes"],
        anchor_sizes=config["model"]["anchor_sizes"],
        anchor_ratios=config["model"]["anchor_ratios"] 
    )

    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    return model

def predict_and_visualize(img_path, model, device, confidence_threshold=0.6):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.inference_mode():
        predictions = model(img_tensor)[0]

    keep_idx = predictions["scores"] >= confidence_threshold
    boxes = predictions["boxes"][keep_idx].cpu().numpy()
    labels = predictions["labels"][keep_idx].cpu().numpy()
    scores = predictions["scores"][keep_idx].cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(img_np)

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        color = "blue" if label == 1 else "red"
        name = "Frame" if label == 1 else "Text"

        rect = patches.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label_text = f"{name}: {score:.2f}"
        ax.text(
            xmin, ymin-5, label_text, color="white", fontsize=10, weight="bold",
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=2)
        )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config_path = "./configs/faster_rcnn_default.yaml"
    weights_path = "./models/faster_rcnn_default_weights.pt"
    img_inference_path = "./data/inference_data/doraemon_2.jpg"
    device = "cpu"

    model = load_trained_model_cpu(config_path, weights_path, device)
    predict_and_visualize(img_inference_path, model, device)
