#!/usr/bin/env python3

import os

import argparse
from pathlib import Path

import math

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling

from src import (
    WeightPruner,
    ActivationsPruner,
    MasksStatisticsCollector,
    WeightsStatisticsCollector,
    summarize_statistics,
    load_shakespeare,
    compute_flops,
    load_config,
    get_device,
    setup_logging,
    setup_wandb,
    auto_configure_pruning,
    print_pruning_schedule,
    validate_pruning_config
)


class PruneCallback(TrainerCallback):
    def __init__(self, pruner):
        self.pruner = pruner


    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)

        self.pruner(state.global_step)


def eval_metric(eval_preds):
    logits, labels = eval_preds.predictions, eval_preds.label_ids

    # Shift logits and labels for causal language modeling
    # Work with numpy arrays directly
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Flatten for loss computation
    shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.reshape(-1)

    # Convert to torch tensors for loss computation
    shift_logits_tensor = torch.from_numpy(shift_logits).float()
    shift_labels_tensor = torch.from_numpy(shift_labels).long()

    # Compute cross entropy loss (ignore padding tokens if any)
    loss = F.cross_entropy(
        shift_logits_tensor,
        shift_labels_tensor,
        ignore_index=-100
    ).item()

    perplexity = math.exp(loss) if loss < 300 else float("inf")
    return {"perplexity": perplexity}


def main():
    parser = argparse.ArgumentParser(description='Train sparse weights model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup wandb with token from .env
    setup_wandb(config)

    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Loaded configuration from {args.config}")

    # Get device
    device = get_device(config['training']['device'])
    logger.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['tokenizer_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader = load_shakespeare(
        config['training']['batch_size'],
        config['training']['seq_len'],
        tokenizer
    )

    # Auto-configure pruning parameters
    config = auto_configure_pruning(config, train_loader)
    print_pruning_schedule(config)

    # Validate pruning configuration
    is_valid, warnings = validate_pruning_config(config)
    if warnings:
        for warning in warnings:
            logger.warning(f"Pruning config warning: {warning}")

    # Initialize model with random weights (not pretrained)
    model_config = GPT2Config.from_pretrained(config['model']['config_name'])
    model = GPT2LMHeadModel(model_config).to(device)
    logger.info(f"Initialized model with random weights using config: {config['model']['config_name']}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps = config['training']['epochs'] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(0.1 * total_steps), # Use separate warmup for LR (10% of total steps is common)
        num_training_steps=total_steps,
    )

    # Initialize pruner with metrics collector
    output_dir = Path(config['paths']['output_dir'])

    callbacks = []
    assert config['mode'] in ['none', 'masked-weights', 'masked-activations'], "Only 'none', 'masked-weights' and 'masked-activations' modes are currently supported"
    if config['mode'] == 'masked-weights':
        callbacks.append(PruneCallback(
            WeightPruner(
                model,
                config['pruning']['target_sparsity'],
                config['pruning']['warmup_steps'],
                config['pruning']['final_prune_step'],
                config['pruning']['prune_freq'],
            )
        ))
    elif config['mode'] == 'masked-activations':
        callbacks.append(PruneCallback(
            ActivationsPruner(
                model,
                config['pruning']['target_sparsity'],
                config['pruning']['warmup_steps'],
                config['pruning']['final_prune_step'],
                config['pruning']['prune_freq'],
            )
        ))

    assert all(callback in ['s-collector', 'm-collector'] for callback in config['training']['callbacks']), "Only 's-collector' and 'm-collector' callbacks are supported"
    s_collector = None
    if 's-collector' in config['training']['callbacks']:
        s_collector = WeightsStatisticsCollector(
            config['collector']['zero_weight_threshold'],
            config['collector']['dead_grad_threshold'],
            config['collector'].get('trackable_weights_layers'),
            config['collector']['s_collect_frequency'],
            config['collector']['dump_frequency'],
            f"./weights_statistics_collector_output/{config['wandb']['project']}",
            config['pruning']['warmup_steps'] + 1
        )
        callbacks.append(s_collector)
    m_collector = None
    if 'm-collector' in config['training']['callbacks']:
        m_collector = MasksStatisticsCollector(
            config['collector'].get('trackable_masks', None),
            config['pruning']['prune_freq'],
            config['collector']['dump_frequency'],
            f"./masks_statistics_collector_output/{config['wandb']['project']}",
            config['pruning']['warmup_steps'] + 1
        )
        callbacks.append(m_collector)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 does not use masked language modeling
    )

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['batch_size'],
            learning_rate=config['training']['lr'],
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            logging_dir=str(output_dir / 'logs'),
            logging_steps=10,
            run_name=config['wandb']['project'],
            save_steps=1000,
            eval_steps=config['training']['eval_steps'],
            eval_strategy='steps',
            save_total_limit=3,
            fp16=config['training'].get('use_bf16', False),
            max_grad_norm=1.0,
            report_to=config['training']['report_to'],
        ),
        data_collator=data_collator,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=eval_metric,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
    ).train()

    try:
        stats = summarize_statistics(m_collector, s_collector)
        run_name = config['wandb']['project']
        out_path = f'{output_dir}/{run_name}/collector_data_summary.pt'
        if not os.path.exists(f'{output_dir}/{run_name}'):
            os.makedirs(f'{output_dir}/{run_name}')
        torch.save(stats, out_path)
    except Exception as e:
        print(f"Error during data collection: {e}")

    flops = compute_flops(model, config['training']['seq_len'], device)
    if flops is not None:
        logger.info(f"Forward pass FLOPs: {flops:,}")

    def infer(model, tokenizer, text):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=50).cpu()
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    infer_result = infer(model, tokenizer, config['training']['infer_text'])
    run_name = config['wandb']['project']
    if not os.path.exists(f'{output_dir}/{run_name}'):
        os.makedirs(f'{output_dir}/{run_name}', exist_ok=True)
    with open(f'{output_dir}/{run_name}/infer_result.txt', 'w') as f:
        f.write(infer_result)

    model.save_pretrained(output_dir)


if __name__ == '__main__':
    main()