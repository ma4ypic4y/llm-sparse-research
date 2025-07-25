import os

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
    ds = load_dataset("text", data_files=data_files)['train'] # , sample_by="document"

    # Process data
    def proccess_text(text):
        # Tokenize the text and split into chunks of seq_len
        text = "\n".join(text['text'] + [tokenizer.eos_token])
        tokenized = tokenizer(text, return_tensors="pt").input_ids[0]
        chunks = list(tokenized.split(seq_len))
        return {'input_ids': chunks}
    # def proccess_text(text):
    #     # Tokenize the text and split into chunks of seq_len
    #     text = [t + tokenizer.eos_token for t in text['text']]
    #     tokenized = tokenizer(text, return_tensors="pt").input_ids[0]
    #     chunks = list(tokenized.split(seq_len))[:-1]
    #     return {'input_ids': chunks}
    # def proccess_text(text):
    #     # Tokenize the text and split into chunks of seq_len
    #     inputs =  tokenizer(text['text'] + tokenizer.eos_token, truncation=True, padding=True, max_length=seq_len)
    #     # inputs['labels'] = inputs['input_ids'].copy()
    #     return inputs

    ds = ds.map(
        proccess_text,
        remove_columns=['text'],
        batched=True,
        batch_size=64,
        # num_proc=16,  # Use multiple processes for faster processing
    ).with_format("torch")

    logger.debug(f"Total examples: {len(ds)}")
    split = ds.train_test_split(test_size=0.1, seed=42)
    logger.debug(f"Train split: {len(split['train'])}, Val split: {len(split['test'])}")

    train_loader = DataLoader(split['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(split['test'], batch_size=batch_size)

    logger.debug(f"DataLoaders created: train={len(train_loader)} batches, val={len(val_loader)} batches")

    return (train_loader, val_loader)

def load_wikitext(batch_size: int, seq_len: int, tokenizer: GPT2TokenizerFast):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "experiments")

    logger = logging.getLogger('sparse_weights.data')

    # check if cache 'wikitext.cache.pt' file exists
    if os.path.exists(os.path.join(cache_dir, 'wikitext.cache.pt')):
        logger.debug("Loading cached WikiText dataset...")
        train = Dataset.load_from_disk(os.path.join(cache_dir, 'wikitext.cache.pt', 'train'))
        test = Dataset.load_from_disk(os.path.join(cache_dir, 'wikitext.cache.pt', 'test'))
    else:
        logger.debug("Cache not found, loading WikiText dataset from scratch...")

        # Load WikiText dataset
        logger.debug("Loading WikiText dataset...")
        train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

        # Process data
        def proccess_text(text):
            # Tokenize the text and split into chunks of seq_len
            inputs =  tokenizer(text['text'], truncation=True, padding=True, max_length=seq_len)
            # inputs['labels'] = inputs['input_ids'].copy()
            return inputs

        train = train.map(
            proccess_text,
            num_proc=16,  # Use multiple processes for faster processing
        ).with_format("torch")
        test = test.map(
            proccess_text,
            num_proc=16,  # Use multiple processes for faster processing
        ).with_format("torch")

        logger.debug(f"Train split: {len(train)}, Val split: {len(test)}")

        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        train.save_to_disk(os.path.join(cache_dir, 'wikitext.cache.pt', 'train'))
        test.save_to_disk(os.path.join(cache_dir, 'wikitext.cache.pt', 'test'))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test, batch_size=batch_size)

    return (train_loader, val_loader)

def load_red_pajama(batch_size: int, seq_len: int, tokenizer: GPT2TokenizerFast):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "experiments")

    logger = logging.getLogger('sparse_weights.data')

    # check if cache 'red_pajama.cache.pt' file exists
    if os.path.exists(os.path.join(cache_dir, 'red_pajama.cache.pt')):
        logger.debug("Loading cached Red Pajama dataset...")
        train = Dataset.load_from_disk(os.path.join(cache_dir, 'red_pajama.cache.pt', 'train'))
        test = Dataset.load_from_disk(os.path.join(cache_dir, 'red_pajama.cache.pt', 'test'))
    else:
        logger.debug("Cache not found, loading Red Pajama dataset from scratch...")

        # Load Red Pajama dataset
        logger.debug("Loading Red Pajama dataset...")
        train = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', trust_remote_code=True)["train"]

        split = train.train_test_split(test_size=0.1, seed=42)
        train = split['train']
        test = split['test']

        # Process data
        def proccess_text(text):
            text = [t + tokenizer.eos_token for t in text['text']]
            tokenized = tokenizer(text).input_ids
            chunks = [
                torch.tensor(ids).split(seq_len)
                for ids in tokenized
            ]
            return sum(chunks, [])

        train = train.map(
            lambda text: {'input_ids': proccess_text(text)},
            remove_columns=['text'],
            batched=True,
            num_proc=4,  # Use multiple processes for faster processing
        ).with_format("torch")
        test = test.map(
            lambda text: {'input_ids': proccess_text(text)},
            remove_columns=['text'],
            batched=True,
            num_proc=4,  # Use multiple processes for faster processing
        ).with_format("torch")

        logger.debug(f"Train split: {len(train)}, Val split: {len(test)}")
        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        train.save_to_disk(os.path.join(cache_dir, 'red_pajama.cache.pt', 'train'))
        test.save_to_disk(os.path.join(cache_dir, 'red_pajama.cache.pt', 'test'))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test, batch_size=batch_size)

    return (train_loader, val_loader)