import math

from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

def configure_optimizer(config, params, data_len: int):
    optimizer = AdamW(
        params,
        lr=config['lr'],
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps = config['epochs'] * data_len
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(0.1 * total_steps), # Use separate warmup for LR (10% of total steps is common)
        num_training_steps=total_steps,
    )

    return optimizer, scheduler