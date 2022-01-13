import logging
import sys

import torch
import yaml


def initialize_logger():
    logger = logging.getLogger("")
    formatter = logging.Formatter('[%(asctime)s] - %(funcName)s - %(message)s', datefmt="%a, %d %b %Y %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, formatter


def load_yaml_config(config_file_path: str) -> dict:
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def set_seeds(seed: int):
    torch.manual_seed(seed)

    # Fix numeric divergence due to bug in Cudnn
    # Also: Seems to search for the best algorithm to use; don't use if the input size changes a lot then it hurts
    # performance
    torch.backends.cudnn.benchmark = True


def get_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    return device
