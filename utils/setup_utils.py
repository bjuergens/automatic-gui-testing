import json
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


def pretty_json(dict_like_object: dict):
    """Taken from https://www.tensorflow.org/tensorboard/text_summaries"""
    json_dict = json.dumps(dict_like_object, indent=2)
    return "".join("\t" + line for line in json_dict.splitlines(True))


def save_yaml_config(save_file_path: str, yaml_config: dict):
    with open(save_file_path, "w") as file:
        yaml.safe_dump(yaml_config, file, default_flow_style=False)


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
