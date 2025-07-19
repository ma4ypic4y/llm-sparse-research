import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def shift_labels(x):
    return x[:, 1:].contiguous()


def loss_fn(logits, labels):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )


def evaluate(model, val_loader: DataLoader, device: str):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            logits = model(ids[:, :-1]).logits
            labels = shift_labels(ids)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    return math.exp(total_loss / total_tokens)