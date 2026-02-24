import torch
import yaml
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_processing.dataset import Manga109Dataset
from detection.detection_model import faster_rcnn

def collate_fn(batch):
    return tuple(zip(*batch))

yaml_path = "./configs/faster_rcnn_default.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_transform = None

train_dataset = Manga109Dataset(
    json_file=config["data"]["train_json"],
    img_dir=config["data"]["img_dir"],
    transforms=train_transform
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["data"]["num_workers"],
    collate_fn=collate_fn
)

model = faster_rcnn(
    num_classes=config["model"]["num_classes"],
    anchor_sizes=config["model"]["anchor_sizes"],
    anchor_ratios=config["model"]["anchor_ratios"]
)

optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=config["training"]["learning_rate"],
    momentum=config["training"]["momentum"],
    weight_decay=config["training"]["weight_decay"],
)

def train_step(train_dataloader, model, device, optimizer, scaler):
    model.train()

    epoch_loss = 0.0
    loop = tqdm(train_dataloader, desc="Training", colour="cyan")
    
    for i, (imgs, targets) in enumerate(loop):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        with torch.amp.autocast("cuda"):
            loss_dict = model(imgs, targets)
        if len(loss_dict) == 0:
            continue
        
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        del imgs, targets, loss_dict

    epoch_loss /= len(train_dataloader)
    return epoch_loss

if __name__ == "__main__":
    scaler = torch.amp.GradScaler("cuda")
    l = train_step(train_dataloader, model, device, optimizer, scaler)