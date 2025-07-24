#!/usr/bin/env python3

import os

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import math

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPTNeoConfig, GPTNeoForCausalLM
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling

from src import (
    WeightPruner,
    ActivationsPruner,
    MasksStatisticsCollector,
    WeightsStatisticsCollector,
    summarize_statistics,
    DataWorker,
    Visualizer,
    load_shakespeare,
    compute_flops,
    load_config,
    get_device,
    setup_logging,
    auto_configure_pruning,
    print_pruning_schedule,
    validate_pruning_config,
    replace_linears_with_pruner
)


class PruneCallback(TrainerCallback):
    def __init__(self, pruner):
        self.pruner = pruner


    def on_step_begin(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)

        self.pruner(state.global_step)

eval_metrics = [] # FIXME: temporary solution to make plots on server

def eval_metric(eval_preds):
    logits, labels = eval_preds.predictions, eval_preds.label_ids

    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    loss = F.cross_entropy(torch.tensor(shift_logits), torch.tensor(shift_labels)).item()

    perplexity = math.exp(loss) if loss < 300 else float("inf")
    eval_metrics.append(perplexity)
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

        # Initialize model
    # Используем предустановленную конфигурацию GPT Neo 125M и модифицируем её
    model_config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")

    # Обновляем параметры из нашей конфигурации
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_position_embeddings = config['training']['seq_len']
    model_config.hidden_size = config['model'].get('hidden_size', model_config.hidden_size)
    model_config.num_heads = config['model'].get('num_heads', model_config.num_heads)

    # Для упрощения оставляем оригинальное количество слоев из предустановленной конфигурации
    # или используем минимальное рабочее количество
    if 'num_layers' in config['model']:
        logger.warning(f"Using original num_layers={model_config.num_layers} instead of configured {config['model']['num_layers']} to avoid attention_types configuration issues")

    model = GPTNeoForCausalLM(model_config).to(device)

    # Initialize pruner with metrics collector
    output_dir = Path(config['paths']['output_dir'])

    callbacks = []
    assert config['mode'] in ['none', 'masked-weights', 'masked-activations', 'masked-activations-layer'], "Only 'none', 'masked-weights', 'masked-activations' and 'masked-activations-layer' modes are currently supported"

    # Если используется режим masked-activations-layer, заменяем все Linear слои на LinearActivationsPruner
    if config['mode'] == 'masked-activations-layer':
        logger.info("Replacing all nn.Linear layers with LinearActivationsPruner...")
        replace_linears_with_pruner(model, config['pruning']['target_sparsity'])
        logger.info("Layer replacement completed.")

    # Optimizer and scheduler (создаем ПОСЛЕ замены слоев для правильных параметров)
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

    run_name = f"llm-sparse-research-{config['mode']}"
    if config['mode'] != 'none':
        run_name += f"-{config['pruning']['target_sparsity']}"

    assert all(callback in ['s-collector', 'm-collector'] for callback in config['training']['callbacks']), "Only 's-collector' and 'm-collector' callbacks are supported"
    s_collector = None
    if 's-collector' in config['training']['callbacks']:
        s_collector = WeightsStatisticsCollector(
            config['collector']['zero_weight_threshold'],
            config['collector']['dead_grad_threshold'],
            config['collector'].get('trackable_weights_layers'),
            config['collector']['s_collect_frequency'],
            config['collector']['dump_frequency'],
            f"./weights_statistics_collector_output/{run_name}",
            config['pruning']['warmup_steps']
        )
        callbacks.append(s_collector)
    m_collector = None
    if 'm-collector' in config['training']['callbacks']:
        m_collector = MasksStatisticsCollector(
            config['collector'].get('trackable_masks', None),
            config['pruning']['prune_freq'],
            config['collector']['dump_frequency'],
            f"./masks_statistics_collector_output/{run_name}",
            config['pruning']['warmup_steps']
        )
        callbacks.append(m_collector)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT Neo does not use masked language modeling
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
            run_name=run_name,
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

    flops = compute_flops(model, config['training']['seq_len'], device)
    if flops is not None:
        logger.info(f"Forward pass FLOPs: {flops:,}")

    model.save_pretrained(output_dir)

    logger.info(f"Everythink done! Making summarization....")

    summary_dir = Path(config['paths']['summary_dir'])
    stats = None
    try:
        stats = summarize_statistics(m_collector, s_collector)
        run_name = run_name
        out_path = f'{summary_dir}/{run_name}/collector_data_summary.pt'
        if not os.path.exists(f'{summary_dir}/{run_name}'):
            os.makedirs(f'{summary_dir}/{run_name}')
        torch.save(stats, out_path)
    except Exception as e:
        logger.error(f"Error during data collection: {e}")

    def infer(model, tokenizer, text):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=50).cpu()
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    infer_result = infer(model, tokenizer, config['training']['infer_text'])
    if not os.path.exists(f'{summary_dir}/{run_name}'):
        os.makedirs(f'{summary_dir}/{run_name}', exist_ok=True)
    with open(f'{summary_dir}/{run_name}/infer_result.txt', 'w') as f:
        f.write(infer_result)

    logger.info(f"Making plots....")

    if stats is not None:
        try:
            plt.plot([i * config['training']['eval_steps'] for i in range(len(eval_metrics))], eval_metrics)
            plt.title("Perplexity")
            plt.xlabel("Step")
            plt.ylabel("Perplexity value")
            plt.gca().set_yscale('log')
            plt.savefig(f"{summary_dir}/{run_name}/perplexity.png", dpi=300, bbox_inches='tight')

            worker = DataWorker().load_stats(stats)
            visualizer = Visualizer(worker)

            if s_collector is not None:
                visualizer.visualize_weights_statistics(sort_by='depth', slice_direction='layers', slice_position=-1)
                plt.savefig(f"{summary_dir}/{run_name}/zero-dead-prune-end.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_weights_statistics(sort_by='depth', slice_direction='layers', slice_position=0)
                plt.savefig(f"{summary_dir}/{run_name}/zero-dead-prune-begin.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_weights_distribution(weights_transform='abs_log', sort_by='layer_type', plot_type='violine', slice_direction='layers', slice_position=-1)
                plt.savefig(f"{summary_dir}/{run_name}/weights-prune-end.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_weights_distribution(weights_transform='abs_log', sort_by='layer_type', plot_type='violine', slice_direction='layers', slice_position=0)
                plt.savefig(f"{summary_dir}/{run_name}/weights-prune-begin.png", dpi=300, bbox_inches='tight')

            if m_collector is not None:
                visualizer.visualize_masks_summary_statistics(sort_by='depth', slice_direction='layers')
                plt.savefig(f"{summary_dir}/{run_name}/masks-summary-layers.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_masks_summary_statistics(sort_by='depth', slice_direction='steps')
                plt.savefig(f"{summary_dir}/{run_name}/masks-summary-steps.png", dpi=300, bbox_inches='tight')

                visualizer.visualize_masks_flick_distribution(weights_transform='norm', sort_by='depth', plot_type='violine')
                plt.savefig(f"{summary_dir}/{run_name}/masks-flick-distribution.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
    else:
        logger.error(f"No stats to make plots")

    logger.info(f"Done!")


if __name__ == '__main__':
    main()