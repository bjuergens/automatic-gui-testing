import logging
import os
from typing import Tuple, Union

import h5py
import torch
from PIL import Image
from torchvision import transforms

from models import select_vae_model, select_rnn_model, Controller
from models.vae import BaseVAE
from utils.setup_utils import load_yaml_config
from utils.constants import (
    GUI_ENV_INITIAL_STATE_FILE_PATH, INITIAL_OBS_LATENT_VECTOR_FILE_NAME, MAX_COORDINATE
)


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


def get_rnn_action_transformation_function(max_coordinate_size_for_task: int, reduce_action_coordinate_space_by: int,
                                           action_transformation_function_type: str):

    if reduce_action_coordinate_space_by > 0:
        already_reduced_factor = MAX_COORDINATE // max_coordinate_size_for_task
        assert isinstance(already_reduced_factor, int), ("For simplicity used_max_coordinate_size must be a multiple "
                                                         f"of MAX_COORDINATE (which is {MAX_COORDINATE})")

        reduce_factor = reduce_action_coordinate_space_by / already_reduced_factor
        # -1.0 because coordinates start at 0
        new_max_coordinate = (max_coordinate_size_for_task / reduce_factor) - 1.0

        rnn_action_transformation_functions = [
            transforms.Lambda(lambda x: torch.div(x, reduce_factor, rounding_mode="floor"))
        ]
    else:
        rnn_action_transformation_functions = []
        new_max_coordinate = max_coordinate_size_for_task - 1.0

    if action_transformation_function_type == "tanh":
        rnn_action_transformation_functions.append(
            transforms.Lambda(lambda x: ((2.0 * x) / new_max_coordinate) - 1.0)
        )

    return transforms.Compose(rnn_action_transformation_functions)


def get_rnn_reward_transformation_function(reward_output_mode: str, reward_output_activation_function: str):
    """
    Reward output mode mse: -> Use [0, 1] or [-1, 1] range depending on output activation function
    Reward output mode bce: -> Only use discrete 0 or 1 reward
    """

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

    return rewards_transformation_function


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
                      rnn_batch_size=None, vae_directory=None):
    config = load_yaml_config(os.path.join(model_dir, "config.yaml"))
    model_name = config["model_parameters"]["name"]

    if model_type == "vae":
        model_class = select_vae_model(model_name)
        model = model_class(config["model_parameters"]).to(device)
    elif model_type == "rnn":
        if rnn_batch_size is None:
            rnn_batch_size = config["experiment_parameters"]["batch_size"]

        vae_config = load_yaml_config(os.path.join(vae_directory, "config.yaml"))
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


def load_rnn_architecture(rnn_directory: str, vae_directory: str, device: torch.device, batch_size=None,
                          load_best: bool = True,
                          load_optimizer: bool = False) -> Union[Tuple[BaseVAE, str], Tuple[BaseVAE, str, dict]]:
    return load_architecture(
        "rnn",
        model_dir=rnn_directory,
        device=device,
        load_best=load_best,
        load_optimizer=load_optimizer,
        rnn_batch_size=batch_size,
        vae_directory=vae_directory
    )


def construct_controller(rnn_dir: str, vae_dir: str):
    rnn_config = load_yaml_config(os.path.join(rnn_dir, "config.yaml"))
    vae_config = load_yaml_config(os.path.join(vae_dir, "config.yaml"))
    latent_size = vae_config["model_parameters"]["latent_size"]
    hidden_size = rnn_config["model_parameters"]["hidden_size"]
    action_size = rnn_config["model_parameters"]["action_size"]

    controller = Controller(latent_size, hidden_size, action_size)

    return controller


def load_controller_parameters(controller, controller_directory: str, device: torch.device):
    state = torch.load(os.path.join(controller_directory, "best.pt"), map_location=device)

    # Take minus of the reward because when saving we "convert" it back to the normal way of summing up the fitness
    # For training we however take the negative amount as the CMA-ES implementation minimizes the fitness instead
    # of maximizing it
    current_best = -state["reward"]
    controller.load_state_dict(state["state_dict"])

    return controller, current_best


def generate_initial_observation_latent_vector(vae_dir, device, load_best: bool = True):
    initial_obs_path = os.path.join(vae_dir, INITIAL_OBS_LATENT_VECTOR_FILE_NAME)

    if os.path.exists(initial_obs_path):
        logging.info("Initial observation in latent space already calculated, continuing")
        return initial_obs_path

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

    # Explicitly delete vae to free memory from gpu
    del vae
    torch.cuda.empty_cache()
    logging.info("Calculated and stored the initial observation in latent space")

    return initial_obs_path
