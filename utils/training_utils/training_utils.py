import os
from typing import Tuple, Union

import h5py
import torch
from PIL import Image
from torchvision import transforms

from models import select_vae_model, select_rnn_model
from models.vae import BaseVAE
from utils.setup_utils import load_yaml_config

GUI_ENV_INITIAL_STATE_FILE_PATH = "res/gui_env_initial_state.png"


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


def rnn_transformation_functions(reward_output_mode: str, reward_output_activation_function: str):
    """
    Reward output mode mse: -> Use [0, 1] or [-1, 1] range depending on output activation function
    Reward output mode bce: -> Only use discrete 0 or 1 reward
    """

    actions_transformation_function = transforms.Lambda(lambda x: ((2.0 * x) / 447.0) - 1.0)

    if reward_output_mode == "mse":
        if reward_output_activation_function == "sigmoid":
            # Rewards are already in [0, 1] range coming from data generation
            rewards_transformation_function = None
        elif reward_output_activation_function == "tanh":
            rewards_transformation_function = transforms.Lambda(lambda x: 2.0 * x - 1.0)
        else:
            raise RuntimeError(f"Reward output activation function '{reward_output_activation_function}' unknown")
    elif reward_output_mode == "bce":
        # Convert rewards > 0 to 1 and rewards equal to 0 remain 0
        rewards_transformation_function = transforms.Lambda(lambda x: x.greater(0).float())
    else:
        raise RuntimeError(f"Reward output mode '{reward_output_mode}' unknown")

    return actions_transformation_function, rewards_transformation_function


def get_dataset_mean_std(dataset: str):
    if dataset == "gui_env_image_dataset_500k_normalize":
        return [0.9338, 0.9313, 0.9288], [0.1275, 0.1329, 0.141]
    else:
        return None, None


def save_checkpoint(state: dict, is_best: bool, checkpoint_filename: str, best_filename: str):
    torch.save(state, checkpoint_filename)
    if is_best:
        torch.save(state, best_filename)


def load_architecture(model_type: str, model_dir: str, device, load_best: bool = True, load_optimizer: bool = False,
                      rnn_batch_size=None):
    config = load_yaml_config(os.path.join(model_dir, "config.yaml"))
    model_name = config["model_parameters"]["name"]

    if model_type == "vae":
        model_class = select_vae_model(model_name)
        model = model_class(config["model_parameters"]).to(device)
    elif model_type == "rnn":
        if rnn_batch_size is None:
            rnn_batch_size = config["experiment_parameters"]["batch_size"]

        vae_config = load_yaml_config(os.path.join(config["vae_parameters"]["directory"], "config.yaml"))
        latent_size = vae_config["model_parameters"]["latent_size"]

        model_class = select_rnn_model(model_name)
        model = model_class(config["model_parameters"], latent_size, rnn_batch_size, device).to(device)
    else:
        raise RuntimeError(f"Model type {model_type} unknown")

    if load_best:
        state_dict_file_name = "best.pt"
    else:
        state_dict_file_name = "checkpoint.pt"

    checkpoint = torch.load(os.path.join(model_dir, state_dict_file_name), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if load_optimizer:
        return model, model_name, checkpoint["optimizer"]
    return model, model_name


def load_vae_architecture(vae_directory: str, device: torch.device, load_best: bool = True,
                          load_optimizer: bool = False) -> Union[Tuple[BaseVAE, str], Tuple[BaseVAE, str, dict]]:
    return load_architecture(
        "vae",
        model_dir=vae_directory,
        device=device,
        load_best=load_best,
        load_optimizer=load_optimizer
    )


def load_rnn_architecture(rnn_directory: str, device: torch.device, batch_size=None, load_best: bool = True,
                          load_optimizer: bool = False) -> Union[Tuple[BaseVAE, str], Tuple[BaseVAE, str, dict]]:
    return load_architecture(
        "rnn",
        model_dir=rnn_directory,
        device=device,
        load_best=load_best,
        load_optimizer=load_optimizer,
        rnn_batch_size=batch_size
    )


def generate_initial_observation_latent_vector(initial_obs_path: str, vae_dir, device, load_best: bool = True):
    vae, _ = load_vae_architecture(vae_dir, device, load_best=load_best, load_optimizer=False)
    vae.eval()

    vae_config = load_yaml_config(os.path.join(vae_dir, "config.yaml"))

    img_size = vae_config["experiment_parameters"]["img_size"]
    dataset = vae_config["experiment_parameters"]["dataset"]
    output_activation_function = vae_config["model_parameters"]["output_activation_function"]

    transformation_functions = vae_transformation_functions(img_size=img_size, dataset=dataset,
                                                            output_activation_function=output_activation_function)

    img = Image.open(GUI_ENV_INITIAL_STATE_FILE_PATH)
    img = transformation_functions(img)
    img = img.unsqueeze(0).to(device)  # Simulate batch dimension

    with torch.no_grad():
        mu, log_var = vae.encode(img)

    with h5py.File(initial_obs_path, "w") as f:
        f.create_dataset(f"mu", data=mu.cpu())
        f.create_dataset(f"log_var", data=log_var.cpu())
