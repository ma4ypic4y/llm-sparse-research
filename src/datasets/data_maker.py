import logging

import torch

from datasets import load_dataset, Dataset

from transformers.tokenization_utils import PreTrainedTokenizer


def load_shakespeare(seq_len: int, tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset]:
    # Process data
    def proccess_text(text):
        # Tokenize the text and split into chunks of seq_len
        # text = "\n".join(text['text'])
        tokenized = tokenizer(text['text'], return_tensors="pt").input_ids[0]
        tokenized = tokenized[:-((tokenized.numel() - 1) % seq_len)]
        input_ids = [
            tokenized[i:i + seq_len] for i in range(0, tokenized.numel() - seq_len)
        ]
        labels = [
            tokenized[i + 1:i + seq_len + 1] for i in range(0, tokenized.numel() - seq_len)
        ]
        return {'input_ids': input_ids, 'labels': labels}
    
    # Load Tiny Shakespeare dataset
    data_files = {"train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"}
    ds = load_dataset("text", data_files=data_files, sample_by="document")['train']

    ds = ds.map(
        proccess_text,
        remove_columns=['text'],
        batched=True,
    ).with_format("torch")

    split = ds.train_test_split(test_size=0.001, seed=42)

    return (split['train'], split['test'])

def load_wikitext(seq_len: int, tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset]:
    # Process data
    def proccess_text(text):
        # Tokenize the text and split into chunks of seq_len
        inputs =  tokenizer(text['text'], truncation=True, padding=True, max_length=seq_len)
        # inputs['labels'] = inputs['input_ids'].copy()
        return inputs
    
    # Load WikiText dataset
    train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    train = train.map(
        proccess_text,
        num_proc=16,  # Use multiple processes for faster processing
    ).with_format("torch")
    test = test.map(
        proccess_text,
        num_proc=16,  # Use multiple processes for faster processing
    ).with_format("torch")

    return (train, test)

def load_red_pajama(seq_len: int, tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset]:
    # Process data
    def proccess_text(text):
        text = [t + tokenizer.eos_token for t in text['text']]
        tokenized = tokenizer(text).input_ids
        chunks = [
            torch.tensor(ids).split(seq_len)
            for ids in tokenized
        ]
        return sum(chunks, [])
    
    # Load Red Pajama dataset
    ds = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', trust_remote_code=True)["train"]
    ds = ds.map(
        lambda text: {'input_ids': proccess_text(text)},
        remove_columns=['text'],
        batched=True,
        num_proc=4,  # Use multiple processes for faster processing
    ).with_format("torch")

    split = ds.train_test_split(test_size=0.1, seed=42)

    return (split['train'], split['test'])

def make_datasets(
    dataset_name: str,
    seq_len: int,
    tokenizer: PreTrainedTokenizer
) -> tuple[Dataset, Dataset]:
    datasets_map = {
        "shakespeare": load_shakespeare,
        "wikitext": load_wikitext,
        "red_pajama": load_red_pajama,
    }

    logger = logging.getLogger('sparse_weights.data')
    logger.debug(f"Loading {dataset_name} dataset...")

    if dataset_name not in datasets_map:
        raise ValueError(f"make_datasets: Unknown dataset {dataset_name}. Available datasets: {list(datasets_map.keys())}")
    
    train, test = datasets_map[dataset_name](seq_len, tokenizer)

    train_examples, test_examples, total_examples = len(train), len(test), len(train) + len(test)
    logger.debug(f"Train split: {train_examples} ({100.0 * train_examples / total_examples:.2%}%), Val split: {test_examples} ({100.0 * test_examples / total_examples:.2%}%), Total examples: {total_examples}")

    return (train, test)
