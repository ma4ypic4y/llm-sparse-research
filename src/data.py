import torch
import logging
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def load_shakespeare(batch_size: int, seq_len: int, tokenizer: GPT2TokenizerFast):
    logger = logging.getLogger('sparse_weights.data')

    # Load Tiny Shakespeare dataset
    logger.debug("Loading Tiny Shakespeare dataset...")
    data_files = {
        "train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    }
    ds = load_dataset("text", data_files=data_files)

    # Split into train/val (90/10)
    full = ds['train']
    logger.debug(f"Total examples: {len(full)}")
    split = full.train_test_split(test_size=0.1, seed=42)
    logger.debug(f"Train split: {len(split['train'])}, Val split: {len(split['test'])}")

    def tokenize(ex):
        # Add EOS token at the end for proper language modeling
        text = ex['text'] + tokenizer.eos_token
        return tokenizer(text, add_special_tokens=False).input_ids

    # Process each split
    for split_name in ['train', 'test']:
        # Tokenization
        split[split_name] = split[split_name].map(
            lambda ex: {'ids': tokenize(ex)},
            remove_columns=['text'],
            num_proc=4
        )

        # Group into sequences of given length
        def group(examples):
            ids = sum(examples['ids'], [])
            n = len(ids) // seq_len * seq_len
            ids = ids[:n]
            return {
                'input_ids': [ids[i:i+seq_len] for i in range(0, n, seq_len)]
            }

        split[split_name] = split[split_name].map(
            group,
            batched=True,
            batch_size=batch_size,
            remove_columns=['ids']
        )
        split[split_name].set_format(type='torch', columns=['input_ids'])

    train_loader = DataLoader(split['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(split['test'], batch_size=batch_size)

    logger.debug(f"DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")

    return (train_loader, val_loader)