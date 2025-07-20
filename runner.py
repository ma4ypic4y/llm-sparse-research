#!/usr/bin/env python3

import os
import sys
import math
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import wandb
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from src import (
    WeightPruner,
    PruningMetricsCollector,
    load_shakespeare,
    shift_labels,
    loss_fn,
    evaluate,
    compute_flops,
    load_config,
    get_device,
    setup_logging,
    auto_configure_pruning,
    print_pruning_schedule,
    validate_pruning_config
)


def main():
    # Load environment variables from .env file
    load_dotenv()

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

    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Loaded configuration from {args.config}")

    # Check wandb token
    wandb_token = os.getenv('WAN_DB_TOKEN')
    if wandb_token:
        os.environ['WANDB_API_KEY'] = wandb_token
        logger.info("Loaded wandb token from .env file")
    else:
        logger.warning("WAN_DB_TOKEN not found in .env file")

    # Initialize wandb
    wandb.init(
        project=config['project_name'],
        config=config
    )
    logger.info(f"Initialized wandb project: {config['project_name']}")

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
    logger.info(f"Data loaded: {len(train_loader)} train and {len(val_loader)} val batches")

    # Auto-configure pruning parameters
    config = auto_configure_pruning(config, train_loader)
    print_pruning_schedule(config)

    # Validate pruning configuration
    is_valid, warnings = validate_pruning_config(config)
    if warnings:
        for warning in warnings:
            logger.warning(f"Pruning config warning: {warning}")

    # Initialize model
    model_config = GPT2Config.from_pretrained(config['model']['config_name'])
    model = GPT2LMHeadModel(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=0.01,      # Add some regularization
        betas=(0.9, 0.999),     # Standard betas for GPT
        eps=1e-8
    )

    total_steps = config['training']['epochs'] * len(train_loader)
    # Use separate warmup for LR (10% of total steps is common)
    lr_warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        lr_warmup_steps,
        total_steps
    )
    logger.info(f"Optimizer and scheduler initialized:")
    logger.info(f"  - Total training steps: {total_steps}")
    logger.info(f"  - LR warmup steps: {lr_warmup_steps}")
    logger.info(f"  - Learning rate: {config['training']['lr']}")
    logger.info(f"  - Weight decay: 0.01")
    logger.info(f"  - Gradient clipping: max_norm=1.0")

    # Initialize mixed precision training
    use_bf16 = config['training'].get('use_bf16', False)
    scaler = GradScaler() if use_bf16 else None
    if use_bf16:
        logger.info("Mixed precision training enabled (bf16)")

    # Initialize metrics collector
    metrics_collector = PruningMetricsCollector()
    logger.info("Metrics collector initialized")

    # Initialize pruner with metrics collector
    pruner = WeightPruner(
        model,
        config['pruning']['target_sparsity'],
        config['pruning']['warmup_steps'],
        config['pruning']['final_prune_step'],
        config['pruning']['prune_freq'],
        metrics_collector=metrics_collector
    )
    logger.info(f"Pruner initialized. Target sparsity: {config['pruning']['target_sparsity']:.1%}")

    # Create output directory
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Main training loop
    global_step = 0
    logger.info(f"Starting training for {config['training']['epochs']} epochs...")

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['training']['epochs']}"
        )

        for batch in progress_bar:
            ids = batch['input_ids'].to(device)

            # Forward pass with mixed precision if enabled
            if use_bf16:
                with autocast(dtype=torch.bfloat16):
                    logits = model(ids[:, :-1]).logits
                    labels = shift_labels(ids)
                    loss = loss_fn(logits, labels)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids[:, :-1]).logits
                labels = shift_labels(ids)
                loss = loss_fn(logits, labels)

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Apply pruning with metrics collection
            pruning_stats = pruner(global_step, model)

            # Logging
            epoch_loss += loss.item()
            num_batches += 1

            # Basic training metrics
            wandb.log({
                'train_loss': loss.item(),
                'train_ppl': math.exp(loss.item()),
                'learning_rate': scheduler.get_last_lr()[0]
            }, step=global_step)

            # Update progress bar with key metrics
            postfix_dict = {
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss.item()):.2f}'
            }

            # Add key pruning metrics to progress bar
            if 'sparsity/overall' in pruning_stats:
                postfix_dict['sparsity'] = f"{pruning_stats['sparsity/overall']:.2%}"
            if 'weight_revival/revival_rate' in pruning_stats:
                postfix_dict['revival'] = f"{pruning_stats['weight_revival/revival_rate']:.3f}"

            progress_bar.set_postfix(postfix_dict)

            global_step += 1

        val_ppl = evaluate(model, val_loader, device)
        avg_train_loss = epoch_loss / num_batches

        # Log additional diagnostics
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            'validation_ppl': val_ppl,
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'current_lr': current_lr
        }, step=global_step)

        logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_ppl={val_ppl:.2f}, lr={current_lr:.6f}")

        # Warning if perplexity is too high
        if val_ppl > 500:
            logger.warning(f"⚠️  High validation perplexity detected: {val_ppl:.1f}")
        elif val_ppl < 10:
            logger.info(f"🎯 Good validation perplexity: {val_ppl:.1f}")

        if hasattr(pruner, 'metrics_collector'):
            epoch_metrics = pruner.metrics_collector.collect_all_metrics(model, global_step)

            if 'sparsity/overall' in epoch_metrics:
                prunable_sparsity = epoch_metrics['sparsity/overall']
                all_params_sparsity = epoch_metrics.get('sparsity/all_parameters', 0)

                logger.info(f"Sparsity - Prunable weights: {prunable_sparsity:.2%}")
                logger.info(f"Sparsity - All parameters: {all_params_sparsity:.2%}")

                if 'stats/total_parameters' in epoch_metrics:
                    total_params = epoch_metrics['stats/total_parameters']
                    prunable_params = epoch_metrics.get('stats/prunable_parameters', 0)
                    prunable_ratio = epoch_metrics.get('stats/prunable_ratio', 0)

                    logger.info(f"Model stats - Total params: {total_params:,}, "
                              f"Prunable: {prunable_params:,} ({prunable_ratio:.1%})")

                if 'weight_revival/revival_rate' in epoch_metrics:
                    revival_rate = epoch_metrics['weight_revival/revival_rate']
                    total_revived = epoch_metrics.get('weight_revival/total_revived', 0)
                    logger.info(f"Weight revival - Rate: {revival_rate:.3f}, "
                              f"Total revived: {total_revived}")
            else:
                logger.info("Current sparsity: metrics not available")

    logger.info("Generating final reports...")

    final_step_metrics = metrics_collector.collect_all_metrics(model, global_step)
    final_metrics = metrics_collector.get_final_report()
    wandb.log(final_metrics)

    pruning_summary = pruner.get_pruning_summary()
    wandb.log(pruning_summary)

    logger.info("="*60)
    logger.info("FINAL SPARSIFICATION REPORT")
    logger.info("="*60)

    prunable_sparsity = final_step_metrics.get('sparsity/overall', 0)
    all_params_sparsity = final_step_metrics.get('sparsity/all_parameters', 0)
    target_sparsity = config['pruning']['target_sparsity']

    logger.info(f"🎯 TARGET SPARSITY: {target_sparsity:.1%}")
    logger.info(f"📊 ACHIEVED SPARSITY:")
    logger.info(f"   - Prunable weights: {prunable_sparsity:.2%}")
    logger.info(f"   - All parameters: {all_params_sparsity:.2%}")
    logger.info(f"   - Efficiency: {prunable_sparsity/target_sparsity:.1%} of target")

    total_params = final_step_metrics.get('stats/total_parameters', 0)
    prunable_params = final_step_metrics.get('stats/prunable_parameters', 0)
    prunable_ratio = final_step_metrics.get('stats/prunable_ratio', 0)

    logger.info(f"🔢 MODEL PARAMETERS:")
    logger.info(f"   - Total: {total_params:,}")
    logger.info(f"   - Prunable: {prunable_params:,} ({prunable_ratio:.1%})")
    logger.info(f"   - Zeroed: {int(prunable_params * prunable_sparsity):,}")

    total_revivals = final_metrics.get('final/total_weight_revivals', 0)
    never_revived_pct = final_metrics.get('final/never_revived_percentage', 0)
    avg_duration = final_metrics.get('final/avg_zero_duration', 0)

    logger.info(f"🔄 WEIGHT REVIVAL ANALYSIS:")
    logger.info(f"   - Total revivals: {total_revivals}")
    if isinstance(never_revived_pct, (int, float)):
        logger.info(f"   - Never revived: {never_revived_pct:.1%}")
        if never_revived_pct > 0.8:
            logger.warning("   ⚠️  WARNING: Very high % of irreversibly zeroed weights!")
        elif never_revived_pct > 0.6:
            logger.info("   ⚠️  High % of stably zeroed weights")
        else:
            logger.info("   ✅ Healthy weight dynamics")
    logger.info(f"   - Average zero duration: {avg_duration:.1f} steps")

    num_prune_steps = pruning_summary.get('num_pruning_steps', 0)
    avg_increment = pruning_summary.get('avg_increment_per_step', 0)

    logger.info(f"✂️  PRUNING STATISTICS:")
    logger.info(f"   - Number of pruning steps: {num_prune_steps}")
    logger.info(f"   - Average increment per step: {avg_increment:.3f}")
    logger.info(f"   - Configuration: warmup={config['pruning']['warmup_steps']}, "
              f"freq={config['pruning']['prune_freq']}, final={config['pruning']['final_prune_step']}")

    logger.info("="*60)

    logger.info(f"Final metrics: {final_metrics}")
    logger.info(f"Pruning summary: {pruning_summary}")

    flops = compute_flops(model, config['training']['seq_len'], device)
    if flops is not None:
        wandb.log({'flops_forward': flops})
        logger.info(f"Forward pass FLOPs: {flops:,}")
    else:
        logger.warning("Could not compute FLOPs (ptflops not available)")

    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)

    if flops is not None:
        (output_dir / 'flops.txt').write_text(str(flops))
    import json
    metrics_summary = {
        'final_metrics': final_metrics,
        'pruning_summary': pruning_summary,
        'config': config
    }

    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Training completed successfully!")
    logger.info(f"Metrics summary saved to {output_dir}/metrics_summary.json")
    wandb.finish()


if __name__ == '__main__':
    main()