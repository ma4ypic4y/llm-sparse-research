from typing import Literal, Tuple

from torch import nn

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

from .nanoGPT import GPT as nanoGPT, GPTConfig as nanoGPTConfig

def make_model(model_name: Literal["gpt-small", "gpt-nano", "gpt-neo", "llama3"], tokenizer_name: Literal["gpt", "llama", "char"] = "gpt", device: str = "cuda") -> Tuple[nn.Module, PreTrainedTokenizer]:
    """Initialize a model and tokenizer based on the specified model name and tokenizer type.
    Args:
        model_name (str): Name of the model to initialize. Options are 'gpt-small', 'gpt-nano', 'gpt-neo', and 'llama3'.
        tokenizer_name (str): Type of tokenizer to use. Options are 'gpt', 'llama', and 'char'.
        device (str): Device to load the model onto, e.g., 'cuda' or 'cpu'.
    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: The initialized model and tokenizer.
    """

    # Initialize tokenizer
    tokenizer_map = {
        "gpt": "gpt2",
        "llama": "meta-llama/Llama-3.2-1B",
        "char": "google/byt5-small"
    }
    if tokenizer_name in tokenizer_map:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_map[tokenizer_name], use_fast=True)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}. Supported tokenizers are 'gpt', 'llama', and 'char'.")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    if model_name == 'gpt-small':
        model_config = GPT2Config.from_pretrained("gpt2")
        model_config.vocab_size = tokenizer.vocab_size
        model = GPT2LMHeadModel(model_config)
    elif model_name == 'gpt-nano':
        model_config = nanoGPTConfig.from_pretrained("gpt2-small")
        model_config.vocab_size = tokenizer.vocab_size
        model = nanoGPT(model_config)
    elif model_name == 'gpt-neo':
        model_config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
        model_config.vocab_size = tokenizer.vocab_size
        model = GPTNeoForCausalLM(model_config)
    elif model_name == 'llama3':
        model_config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        model_config.vocab_size = tokenizer.vocab_size
        model = LlamaForCausalLM(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models are 'gpt-small', 'gpt-nano', 'gpt-neo' and 'llama3'.")
    
    return model.to(device=device), tokenizer