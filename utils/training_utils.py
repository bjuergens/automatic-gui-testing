import os
from typing import Tuple

import torch
from torchvision import transforms

from models import select_vae_model
from models.vae import BaseVAE
from utils.setup_utils import load_yaml_config


def vae_transformation_functions(img_size: int):
    transformation_functions = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    return transformation_functions


def save_checkpoint(state: dict, is_best: bool, checkpoint_filename: str, best_filename: str):
    torch.save(state, checkpoint_filename)
    if is_best:
        torch.save(state, best_filename)


def load_vae_architecture(vae_directory: str, device: torch.device, load_best: bool = True) -> Tuple[BaseVAE, str]:
    vae_config = load_yaml_config(os.path.join(vae_directory, "config.yaml"))
    vae_name = vae_config["model_parameters"]["name"]

    use_kld_warmup = vae_config["experiment_parameters"]["kld_warmup"]
    kld_weight = vae_config["experiment_parameters"]["kld_weight"]

    vae_model = select_vae_model(vae_name)
    vae = vae_model(vae_config["model_parameters"], use_kld_warmup, kld_weight).to(device)

    if load_best:
        state_dict_file_name = "best.pt"
    else:
        state_dict_file_name = "checkpoint.pt"

    checkpoint = torch.load(os.path.join(vae_directory, state_dict_file_name), map_location=device)
    vae.load_state_dict(checkpoint["state_dict"])

    return vae
