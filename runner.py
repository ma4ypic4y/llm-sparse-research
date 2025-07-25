#!/usr/bin/env python3

import os

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import math

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, GPT2Tokenizer, GPTNeoConfig, GPTNeoForCausalLM, GPT2Config, GPT2LMHeadModel
from transformers.trainer_pt_utils import LabelSmoother

from src import (
    PruneCallback,
    make_prune_strategy,
    PerplexityEvaluator,
    Summarize,
    infer_model,
    MasksStatisticsCollector,
    WeightsStatisticsCollector,
    summarize_statistics,
    DataWorker,
    Visualizer,
    load_shakespeare,
    load_wikitext,
    load_red_pajama,
    compute_flops,
    load_config,
    get_device,
    setup_logging,
    auto_configure_pruning,
    print_pruning_schedule,
    validate_pruning_config,
    configure_optimizer,
    nanoGPT,
    nanoGPTConfig
)


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
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    logger.info("Loading data...")
    assert config['training']['dataset'] in ['shakespeare', 'wikitext', 'red_pajama'], "Only 'shakespeare', 'wikitext' and 'red_pajama' datasets are currently supported"
    if config['training']['dataset'] == 'shakespeare':
        train_loader, val_loader = load_shakespeare(
            config['training']['batch_size'],
            config['training']['seq_len'],
            tokenizer
        )
    elif config['training']['dataset'] == 'wikitext':
        train_loader, val_loader = load_wikitext(
            config['training']['batch_size'],
            config['training']['seq_len'],
            tokenizer
        )
    elif config['training']['dataset'] == 'red_pajama':
        train_loader, val_loader = load_red_pajama(
            config['training']['batch_size'],
            config['training']['seq_len'],
            tokenizer
        )

    # Auto-configure pruning parameters
    config = auto_configure_pruning(config, len(train_loader))
    print_pruning_schedule(config)

    # Validate pruning configuration
    is_valid, warnings = validate_pruning_config(config)
    if warnings:
        for warning in warnings:
            logger.warning(f"Pruning config warning: {warning}")

    # Initialize model
    assert config['model']['config_name'] in ['gpt2', 'nanoGPT', 'neoGPT'], "Only 'gpt2', 'nanoGPT' and 'neoGPT' models are currently supported"
    if config['model']['config_name'] == 'gpt2':
        model_config = GPT2Config()
        model = GPT2LMHeadModel(model_config).to(device)
    elif config['model']['config_name'] == 'nanoGPT':
        model_config = nanoGPTConfig()
        model = nanoGPT(model_config).to(device)
    elif config['model']['config_name'] == 'neoGPT':
        model_config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
        model = GPTNeoForCausalLM(model_config).to(device)

    # Initialize pruner with metrics collector
    output_dir = Path(config['paths']['output_dir'])

    callbacks = []

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

    run_name = f"{config['model']['config_name']}-{config['training']['dataset']}-{config['mode']}"
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 does not use masked language modeling
    )

    evaluator = PerplexityEvaluator()

    optimizer = configure_optimizer(config['training'], model.parameters(), len(train_loader))
    label_smoother = LabelSmoother(epsilon=0.0)
    loss_fn = lambda outputs, labels, num_items_in_batch: label_smoother(outputs, labels)

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
            save_steps=99999,
            save_strategy="no",
            eval_steps=config['training']['eval_steps'],
            eval_strategy='steps',
            save_total_limit=3,
            fp16=config['training'].get('use_bf16', False),
            max_grad_norm=1.0,
            report_to=config['training']['report_to'],
            # no_cuda=True,
        ),
        data_collator=data_collator,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=evaluator.evaluate,
        # compute_loss_func=loss_fn,
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