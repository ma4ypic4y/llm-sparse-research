import torch
import logging
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from transformers import GPT2TokenizerFast


def load_shakespeare(batch_size: int, seq_len: int, tokenizer: GPT2TokenizerFast):
    logger = logging.getLogger('sparse_weights.data')

    # Load Tiny Shakespeare dataset
    logger.debug("Loading Tiny Shakespeare dataset...")
    data_files = {
        "train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    }
    ds = load_dataset("text", data_files=data_files, sample_by="document")['train']

    # Process data
    def proccess_text(text):
        # Tokenize the text and split into chunks of seq_len
        text = [t + tokenizer.eos_token for t in text['text']]
        tokenized = tokenizer(text, return_tensors="pt").input_ids[0]
        chunks = list(tokenized.split(seq_len))[:-1]
        return chunks

    ds = ds.map(
        lambda text: {'input_ids': proccess_text(text)},
        remove_columns=['text'],
        batched=True,
    ).with_format("torch")

    logger.debug(f"Total examples: {len(ds)}")
    split = ds.train_test_split(test_size=0.1, seed=42)
    logger.debug(f"Train split: {len(split['train'])}, Val split: {len(split['test'])}")

    train_loader = DataLoader(split['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(split['test'], batch_size=batch_size)

    logger.debug(f"DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")

    return (train_loader, val_loader)