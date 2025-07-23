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

    # Get the full text and tokenize it all at once
    full_text = "\n".join(ds['train']['text'])
    logger.debug(f"Total text length: {len(full_text)} characters")

    # Tokenize the entire text
    all_tokens = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")['input_ids'][0]
    logger.debug(f"Total tokens: {len(all_tokens)}")

    # Split into train/val (90/10)
    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    logger.debug(f"Train tokens: {len(train_tokens)}, Val tokens: {len(val_tokens)}")

    def create_sequences(tokens, seq_len):
        """Create overlapping sequences from tokens"""
        sequences = []
        for i in range(0, len(tokens) - seq_len + 1, seq_len):
            sequences.append(tokens[i:i+seq_len].tolist())
        return sequences

    # Create sequences
    train_sequences = create_sequences(train_tokens, seq_len)
    val_sequences = create_sequences(val_tokens, seq_len)

    logger.debug(f"Created {len(train_sequences)} train sequences, {len(val_sequences)} val sequences")

    # Create datasets
    from datasets import Dataset
    train_dataset = Dataset.from_dict({'input_ids': train_sequences})
    val_dataset = Dataset.from_dict({'input_ids': val_sequences})

    train_dataset.set_format(type='torch', columns=['input_ids'])
    val_dataset.set_format(type='torch', columns=['input_ids'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    logger.debug(f"DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")

    return (train_loader, val_loader)