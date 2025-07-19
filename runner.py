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
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

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
    tokenizer = GPT2TokenizerFast.from_pretrained(config['model']['tokenizer_name'])
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
    optimizer = AdamW(model.parameters(), lr=config['training']['lr'])

    total_steps = config['training']['epochs'] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        config['pruning']['warmup_steps'],
        total_steps
    )
    logger.info(f"Optimizer and scheduler initialized. Total training steps: {total_steps}")

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
            logits = model(ids[:, :-1]).logits
            labels = shift_labels(ids)
            loss = loss_fn(logits, labels)

            loss.backward()
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

        # Validation
        val_ppl = evaluate(model, val_loader, device)
        avg_train_loss = epoch_loss / num_batches

        wandb.log({
            'validation_ppl': val_ppl,
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss
        }, step=global_step)

        logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_ppl={val_ppl:.2f}")

        # Log epoch-level metrics summary
        if hasattr(pruner, 'metrics_collector'):
            epoch_metrics = pruner.metrics_collector.collect_all_metrics(model, global_step)

            # Детальное логирование спарсности
            if 'sparsity/overall' in epoch_metrics:
                prunable_sparsity = epoch_metrics['sparsity/overall']
                all_params_sparsity = epoch_metrics.get('sparsity/all_parameters', 0)

                logger.info(f"Sparsity - Prunable weights: {prunable_sparsity:.2%}")
                logger.info(f"Sparsity - All parameters: {all_params_sparsity:.2%}")

                # Показываем также статистику параметров
                if 'stats/total_parameters' in epoch_metrics:
                    total_params = epoch_metrics['stats/total_parameters']
                    prunable_params = epoch_metrics.get('stats/prunable_parameters', 0)
                    prunable_ratio = epoch_metrics.get('stats/prunable_ratio', 0)

                    logger.info(f"Model stats - Total params: {total_params:,}, "
                              f"Prunable: {prunable_params:,} ({prunable_ratio:.1%})")

                # Показываем revival метрики если есть
                if 'weight_revival/revival_rate' in epoch_metrics:
                    revival_rate = epoch_metrics['weight_revival/revival_rate']
                    total_revived = epoch_metrics.get('weight_revival/total_revived', 0)
                    logger.info(f"Weight revival - Rate: {revival_rate:.3f}, "
                              f"Total revived: {total_revived}")
            else:
                logger.info("Current sparsity: metrics not available")

    # Final metrics and reports
    logger.info("Generating final reports...")

    # Собираем финальные метрики
    final_step_metrics = metrics_collector.collect_all_metrics(model, global_step)

    # Get final metrics report
    final_metrics = metrics_collector.get_final_report()
    wandb.log(final_metrics)

    # Get pruning summary
    pruning_summary = pruner.get_pruning_summary()
    wandb.log(pruning_summary)

    # Детальный финальный отчет
    logger.info("="*80)
    logger.info("ФИНАЛЬНЫЙ ОТЧЕТ ПО СПАРСИФИКАЦИИ")
    logger.info("="*80)

    # Спарсность
    prunable_sparsity = final_step_metrics.get('sparsity/overall', 0)
    all_params_sparsity = final_step_metrics.get('sparsity/all_parameters', 0)
    target_sparsity = config['pruning']['target_sparsity']

    logger.info(f"🎯 ЦЕЛЕВАЯ СПАРСНОСТЬ: {target_sparsity:.1%}")
    logger.info(f"📊 ДОСТИГНУТАЯ СПАРСНОСТЬ:")
    logger.info(f"   - Прунимые веса: {prunable_sparsity:.2%}")
    logger.info(f"   - Все параметры: {all_params_sparsity:.2%}")
    logger.info(f"   - Эффективность: {prunable_sparsity/target_sparsity:.1%} от цели")

    # Параметры модели
    total_params = final_step_metrics.get('stats/total_parameters', 0)
    prunable_params = final_step_metrics.get('stats/prunable_parameters', 0)
    prunable_ratio = final_step_metrics.get('stats/prunable_ratio', 0)

    logger.info(f"🔢 ПАРАМЕТРЫ МОДЕЛИ:")
    logger.info(f"   - Общее количество: {total_params:,}")
    logger.info(f"   - Прунимые: {prunable_params:,} ({prunable_ratio:.1%})")
    logger.info(f"   - Зануленные: {int(prunable_params * prunable_sparsity):,}")

    # Анализ разморозки весов
    total_revivals = final_metrics.get('final/total_weight_revivals', 0)
    never_revived_pct = final_metrics.get('final/never_revived_percentage', 0)
    avg_duration = final_metrics.get('final/avg_zero_duration', 0)

    logger.info(f"🔄 АНАЛИЗ РАЗМОРОЗКИ ВЕСОВ:")
    logger.info(f"   - Общее количество разморозок: {total_revivals}")
    if isinstance(never_revived_pct, (int, float)):
        logger.info(f"   - Никогда не размораживались: {never_revived_pct:.1%}")
        if never_revived_pct > 0.8:
            logger.warning("   ⚠️  ВНИМАНИЕ: Очень высокий % необратимо зануленных весов!")
        elif never_revived_pct > 0.6:
            logger.info("   ⚠️  Высокий % стабильно зануленных весов")
        else:
            logger.info("   ✅ Здоровая динамика весов")
    logger.info(f"   - Средняя длительность зануления: {avg_duration:.1f} шагов")

    # Прунинг статистика
    num_prune_steps = pruning_summary.get('num_pruning_steps', 0)
    avg_increment = pruning_summary.get('avg_increment_per_step', 0)

    logger.info(f"✂️  СТАТИСТИКА ПРУНИНГА:")
    logger.info(f"   - Количество шагов прунинга: {num_prune_steps}")
    logger.info(f"   - Средний прирост за шаг: {avg_increment:.3f}")
    logger.info(f"   - Конфигурация: warmup={config['pruning']['warmup_steps']}, "
              f"freq={config['pruning']['prune_freq']}, final={config['pruning']['final_prune_step']}")

    logger.info("="*80)

    logger.info(f"Final metrics: {final_metrics}")
    logger.info(f"Pruning summary: {pruning_summary}")

    # Compute and log FLOPs
    flops = compute_flops(model, config['training']['seq_len'], device)
    if flops is not None:
        wandb.log({'flops_forward': flops})
        logger.info(f"Forward pass FLOPs: {flops:,}")
    else:
        logger.warning("Could not compute FLOPs (ptflops not available)")

    # Save model and results
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)

    if flops is not None:
        (output_dir / 'flops.txt').write_text(str(flops))

    # Save metrics summary
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