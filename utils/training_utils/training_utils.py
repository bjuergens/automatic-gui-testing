import os
from typing import Tuple, Union

import torch
from torchvision import transforms

from models import select_vae_model
from models.vae import BaseVAE
from utils.setup_utils import load_yaml_config


def vae_transformation_functions(img_size: int, dataset: str, output_activation_function: str):

    if dataset == "gui_env_image_dataset_500k_normalize":
        mean, std = get_dataset_mean_std(dataset)
        transformation_functions = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return transformation_functions

    if output_activation_function == "sigmoid":
        transformation_functions = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    elif output_activation_function == "tanh":
        transformation_functions = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Transforms images to [0, 1] range
            transforms.Lambda(lambda x: 2.0 * x - 1.0)  # Transforms tensors to [-1, 1] range
        ])
    else:
        raise RuntimeError(f"Output activation function {output_activation_function} unknown")

    return transformation_functions


def get_dataset_mean_std(dataset: str):
    if dataset == "gui_env_image_dataset_500k_normalize":
        return [0.9338, 0.9313, 0.9288], [0.1275, 0.1329, 0.141]
    else:
        return None, None


def save_checkpoint(state: dict, is_best: bool, checkpoint_filename: str, best_filename: str):
    torch.save(state, checkpoint_filename)
    if is_best:
        torch.save(state, best_filename)


def load_vae_architecture(vae_directory: str, device: torch.device, load_best: bool = True,
                          load_optimizer: bool = False) -> Union[Tuple[BaseVAE, str], Tuple[BaseVAE, str, dict]]:
    vae_config = load_yaml_config(os.path.join(vae_directory, "config.yaml"))
    vae_name = vae_config["model_parameters"]["name"]

    vae_model = select_vae_model(vae_name)
    vae = vae_model(vae_config["model_parameters"]).to(device)

    if load_best:
        state_dict_file_name = "best.pt"
    else:
        state_dict_file_name = "checkpoint.pt"

    checkpoint = torch.load(os.path.join(vae_directory, state_dict_file_name), map_location=device)
    vae.load_state_dict(checkpoint["state_dict"])

    if load_optimizer:
        return vae, vae_name, checkpoint["optimizer"]
    return vae, vae_name
