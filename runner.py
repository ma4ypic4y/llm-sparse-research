#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollatorForLanguageModeling

from src.pruning import auto_configure_pruning, PruneCallback, make_prune_strategy, print_pruning_schedule, validate_pruning_config
from src.hooks import MasksStatisticsCollector, WeightsStatisticsCollector
from src.training import configure_optimizer, PerplexityEvaluator, Summarize, infer_model
from src.models import make_model
from src.datasets import make_datasets
from src import compute_flops, get_device, setup_logging, load_config


class InferHook(TrainerCallback):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        prompt = "\n"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=500, temperature=0.8, top_k=200)
        print(self.tokenizer.decode(output[0], skip_special_tokens=True))


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

    # Initialize tokenizer
    logger.info(f"Initializing tokenizer ({config['model']['tokenizer_name']}) and model ({config['model']['model_name']}) on {device}...")
    model, tokenizer = make_model(
        config['model']['model_name'],
        config['model']['tokenizer_name'],
        device
    )

    # Load data
    train_dataset, val_dataset = make_datasets(
        config['training']['dataset'],
        config['training']['seq_len'],
        tokenizer
    )
    total_steps_per_epoch = len(train_dataset) // config['training']['batch_size']

    # Auto-configure pruning parameters
    config = auto_configure_pruning(config, total_steps_per_epoch)
    if config['mode'] != 'none':
        print_pruning_schedule(config)
    else:
        print("No pruning schedule to display.")

    # Validate pruning configuration
    is_valid, warnings = validate_pruning_config(config)
    if warnings:
        for warning in warnings:
            logger.warning(f"Pruning config warning: {warning}")

    # Initialize pruner with metrics collector
    output_dir = Path(config['paths']['output_dir'])

    callbacks = [InferHook(model, tokenizer, device)]

    mode_to_prune_strategy = {
        'none': 'none',
        'masked-weights': 'global-weight',
        'masked-activations': 'layer-activation-v1',
        'masked-activations-layer': 'layer-activation-v2',
    }
    assert config['mode'] in list(mode_to_prune_strategy.keys()), f"Only {list(mode_to_prune_strategy.keys())} modes are currently supported"
    callbacks.append(PruneCallback(
        model,
        make_prune_strategy(mode_to_prune_strategy[config['mode']]),
        config['pruning']['target_sparsity'],
        config['pruning']['warmup_steps'],
        config['pruning']['final_prune_step'],
        config['pruning']['prune_freq'],
    ))

    run_name = f"{config['model']['model_name']}-{config['training']['dataset']}-{config['mode']}"
    if config['mode'] != 'none':
        run_name += f"-{config['pruning']['target_sparsity']}"

    assert all(callback in ['s-collector', 'm-collector'] for callback in config['training']['callbacks']), "Only 's-collector' and 'm-collector' callbacks are supported"
    s_collector = None
    if 's-collector' in config['training']['callbacks']:
        s_collector = WeightsStatisticsCollector(
            config['collector']['zero_weight_threshold'],
            config['collector']['dead_grad_threshold'],
            False,
            config['collector'].get('trackable_weights_layers'),
            config['collector']['s_collect_frequency'],
            config['collector']['dump_frequency'],
            f"./weights_statistics_collector_output/{run_name}",
            config['pruning']['warmup_steps'] + 6
        )
        callbacks.append(s_collector)
    m_collector = None
    if 'm-collector' in config['training']['callbacks'] and config['mode'] in ['masked-weights']:
        m_collector = MasksStatisticsCollector(
            config['collector'].get('trackable_masks', None),
            config['pruning']['prune_freq'],
            config['collector']['dump_frequency'],
            f"./masks_statistics_collector_output/{run_name}",
            config['pruning']['warmup_steps']
        )
        callbacks.append(m_collector)

    evaluator = PerplexityEvaluator()

    optimizer = configure_optimizer(model.parameters(), total_steps_per_epoch * config['training']['epochs'], config['training']['lr'])

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            run_name=run_name,
            report_to=config['training']['report_to'],
            logging_dir=str(output_dir / 'logs'),
            logging_steps=10,

            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
            save_strategy="no",
            eval_steps=config['training']['eval_steps'],
            eval_strategy='steps',
            fp16=config['training'].get('use_bf16', False),
            max_grad_norm=1.0,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=evaluator.evaluate,
        optimizers=optimizer,
        callbacks=callbacks,
    ).train()

    flops = compute_flops(model, config['training']['seq_len'], device)
    if flops is not None:
        logger.info(f"Forward pass FLOPs: {flops:,}")

    # model.save_pretrained(output_dir)

    logger.info(f"Everythink done! Making summarization....")

    summary_dir = Path(config['paths']['summary_dir'])

    summarizer = Summarize(f'{summary_dir}/{run_name}', s_collector, m_collector, evaluator)
    summarizer.make_summarization('collector_data_summary.pt')

    infer_result = infer_model(model, tokenizer, config['training']['infer_text'])
    if not os.path.exists(f'{summary_dir}/{run_name}'):
        os.makedirs(f'{summary_dir}/{run_name}', exist_ok=True)
    with open(f'{summary_dir}/{run_name}/infer_result.txt', 'w') as f:
        f.write(infer_result)

    logger.info(f"Making plots....")
    summarizer.make_plots()

    logger.info(f"Done!")


if __name__ == '__main__':
    main()