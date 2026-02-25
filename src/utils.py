import gc
import ctypes
import torch

def clean_ram():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()

def save_checkpoint(model, optimizer, scheduler, results, epoch, curr_best_metric, checkpoint_save_path):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "results": results,
        "epoch": epoch,
        "curr_best_metric": curr_best_metric
    }
    torch.save(obj=checkpoint, f=checkpoint_save_path)

def load_checkpoint(model, optimizer, scheduler, device, checkpoint_save_path):
    checkpoint = torch.load(f=checkpoint_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["results"], checkpoint["epoch"], checkpoint["curr_best_metric"] 