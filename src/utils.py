import yaml
import torch
import logging

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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