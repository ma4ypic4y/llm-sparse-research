import math

import torch
from torch.optim import AdamW

from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

def configure_optimizer(params, steps=3_000, lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999)):
    optimizer = AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=1e-8
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(0.1 * steps), # Use separate warmup for LR (10% of total steps is common)
        num_training_steps=steps,
    )

    return optimizer, scheduler
