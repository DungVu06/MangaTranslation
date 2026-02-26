import torch
import yaml
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision
from albumentations.pytorch import ToTensorV2
from src.data_processing.dataset import Manga109Dataset

yaml_path = "./configs/faster_rcnn_default.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

test_transform = A.Compose([
    ToTensorV2()
], bbox_params=A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
))

test_data = Manga109Dataset(
    json_file=config["data"]["val_json"],
    img_dir=config["data"]["img_dir"],
    transforms=test_transform
)

def visualize_results(samples):
    for sample in samples:
        img = sample["img"]
        targets = sample["targets"]
        preds = sample["preds"]
        
        img_np = img.permute(1, 2, 0).numpy()

        keep_idx = preds["scores"] > 0.5
        pred_boxes = preds["boxes"][keep_idx].numpy()
        pred_labels = preds["labels"][keep_idx].numpy()

        gt_boxes = targets["boxes"].numpy()
        gt_labels = targets["labels"].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Ground Truth", color="green", weight="bold")
        axes[0].axis("off")
        for box, label in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box
            color = "blue" if label == 1 else "red"
            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            axes[0].add_patch(rect)

        axes[1].imshow(img_np)
        axes[1].set_title(f"Prediction", color="orange", weight="bold")
        axes[1].axis("off")
        for box, label in zip(pred_boxes, pred_labels):
            xmin, ymin, xmax, ymax = box
            color = "blue" if label == 1 else "red"
            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            axes[1].add_patch(rect)


def test_step(test_dataloader, model, device):
    model.eval()
    mAP = MeanAveragePrecision()
    hist = []
    loop = tqdm(test_dataloader, desc="Evaluating on test set...", colour="purple", total=len(test_dataloader))

    with torch.inference_mode():
        for imgs, targets in loop:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            outputs = model(imgs)
            targets = [{k: v.to("cpu") for k, v in target.items()} for target in targets]
            outputs = [{k: v.to("cpu") for k, v in output.items()} for output in outputs]

            mAP.update(outputs, targets)

            for img, target, pred in zip(imgs, targets, outputs):
                num_pred_confident = (pred["scores"] > 0.5).sum().item()
                num_gt = len(target["boxes"])
                error_score = abs(num_gt - num_pred_confident)

                if error_score > 0:
                    hist.append({
                        "img": img.cpu(),
                        "targets": target,
                        "preds": pred,
                        "error_score": error_score
                    })
            del imgs, targets, outputs
        
    results = mAP.compute()
    mAP_50_95 = results["map"].item()
    
    return mAP_50_95, hist