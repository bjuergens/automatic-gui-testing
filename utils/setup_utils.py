import json
import logging
import os.path
import random
import sys

import numpy as np
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
    random.seed(seed)
    np.random.seed(seed)

    # Fix numeric divergence due to bug in Cudnn
    # Also: Seems to search for the best algorithm to use; don't use if the input size changes a lot then it hurts
    # performance
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)


def get_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    return device


def get_depending_model_path(model_type: str, model_dir: str):
    config = load_yaml_config(os.path.join(model_dir, "config.yaml"))

    if model_type == "rnn":
        return config["vae_parameters"]["directory"]
    elif model_type == "controller":
        return config["rnn_parameters"]["rnn_dir"]
    else:
        raise RuntimeError(f"Model type '{model_type}' unknown")


def resolve_model_path(path_to_resolve: str, model_copied: bool, location: str):
    """
    Resolve model paths

    Model paths are stored as relative paths in the config.yaml files in the trained model directories. If they are all
    on the same server (for example after training) then nothing has to be done, but it can be that for example the M
    model shall be visualized on a local laptop. Then the V model can be copied to the local laptop and the M model
    can remain on the server, while the root directory on the server is mounted locally using sshfs. Mounting has to
    happen in a folder in the home-directory where the location parameter of this function is the directory name.
    """

    if location == "local":
        return path_to_resolve

    if model_copied:
        prepend_path = f"logs/{location}"
    else:
        # Assume that the root folder of this project is mounted in the home directory under 'location' using sshfs
        prepend_path = f"{os.path.expanduser('~')}/{location}"

    return os.path.join(prepend_path, path_to_resolve)
