import yaml
import torch
import logging
import os
from dotenv import load_dotenv

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: dict) -> None:
    """
    Setup Weights & Biases with token from .env file
    """
    # Load environment variables from .env file
    load_dotenv('../.env', override=True)

    # Check if wandb is enabled in config
    if not config.get('wandb', {}).get('enabled', False):
        return

    # Get wandb token from environment
    wandb_token = os.getenv('WANDB_TOKEN')
    if wandb_token:
        os.environ['WANDB_API_KEY'] = wandb_token
        print(f"WANDB_TOKEN loaded from .env file")
    else:
        print("Warning: WANDB_TOKEN not found in .env file. Wandb will use default authentication.")

    # Set wandb project name
    project_name = config.get('wandb', {}).get('project', 'sparse-weights')
    os.environ['WANDB_PROJECT'] = project_name


def compute_flops(model, seq_len: int, device: str):
    if get_model_complexity_info is None:
        return None

    macs, _ = get_model_complexity_info(
        model, (seq_len,),
        as_strings=False,
        input_constructor=lambda _: torch.ones(
            (1, seq_len), dtype=torch.long, device=device
        ),
        verbose=False
    )
    return 2 * macs


def get_device(preferred_device: str = 'cuda') -> str:
    if preferred_device == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def setup_logging(config: dict) -> logging.Logger:
    log_config = config.get('logging', {})

    # Setup logging level
    level = getattr(logging, log_config.get('level', 'INFO').upper())

    # Setup format
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Get root logger
    logger = logging.getLogger('sparse_weights')
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Setup file handler (if specified)
    log_file = log_config.get('file')
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger