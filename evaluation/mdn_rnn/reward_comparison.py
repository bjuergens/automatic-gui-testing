import logging
import os
from typing import List, Dict, Tuple

import click
import numpy as np
import torch
from tqdm import tqdm

from data.dataset_implementations import get_main_rnn_data_loader
from envs.simulated_gui_env import SimulatedGUIEnv
from models import select_rnn_model
from utils.data_processing_utils import get_vae_preprocessed_data_path_name
from utils.setup_utils import (
    initialize_logger, get_depending_model_path, resolve_model_path, get_device, load_yaml_config
)
from utils.training_utils.training_utils import (
    generate_initial_observation_latent_vector, get_rnn_action_transformation_function,
    get_rnn_reward_transformation_function
)


def compare_reward_of_m_model_to_sequence(
        dict_of_sequence_actions: Dict[int, List[List[Tuple[torch.Tensor, torch.Tensor]]]], rnn_dir: str,
        vae_dir: str, max_coordinate_size_for_task: int, device: torch.device, initial_obs_path: str,
        temperature: float = 1.0, load_best_rnn: bool = True, render: bool = False
) -> Dict[int, List[float]]:

    env = SimulatedGUIEnv(
        rnn_dir=rnn_dir,
        vae_dir=vae_dir,
        initial_obs_path=initial_obs_path,
        max_coordinate_size_for_task=max_coordinate_size_for_task,
        temperature=temperature,
        device=device,
        load_best_rnn=load_best_rnn,
        render=render
    )

    total_iterations = sum([seq_len * len(seq_list) for seq_len, seq_list in dict_of_sequence_actions.items()])
    progress_bar = tqdm(total=total_iterations, desc="Reward Comparison of sequences and trained M model")

    sequence_rewards = {}

    for sequence_length, list_of_sequence_actions in dict_of_sequence_actions.items():
        for sequence_actions in list_of_sequence_actions:
            assert len(sequence_actions.size()) == 2 and sequence_actions.size(1) == 2
            env.reset()

            summed_reward = 0.0

            for action in sequence_actions:
                _, rew, _, _ = env.step((action[0], action[1]))
                summed_reward += rew
                progress_bar.update(1)

            try:
                sequence_rewards[sequence_length]
            except KeyError:
                sequence_rewards[sequence_length] = [summed_reward]
            else:
                sequence_rewards[sequence_length].append(summed_reward)

    progress_bar.close()

    return sequence_rewards


def start_reward_comparison(rnn_dir, vae_dir, val_dataset, model_name, reward_transformation_function, device,
                            temperature: float = 1.0, load_best_rnn: bool = True):
    validation_sequences = val_dataset.get_validation_sequences_for_m_model_comparison()

    dict_of_sequence_actions = {}
    dict_of_sequence_rewards = {}
    for sequence_length, sequence_list in validation_sequences.items():
        dict_of_sequence_actions[sequence_length] = [seq.actions for seq in sequence_list]

        if model_name == "lstm_bce":
            # Compare also to 0 _or_ 1 rewards because the LSTMBCE model predicts only 0 or 1
            dict_of_sequence_rewards[sequence_length] = [
                reward_transformation_function(seq.rewards).sum() for seq in sequence_list
            ]
        else:
            dict_of_sequence_rewards[sequence_length] = [seq.rewards.sum() for seq in sequence_list]

    initial_obs_path = generate_initial_observation_latent_vector(vae_dir, device, load_best=True)

    compared_rewards = compare_reward_of_m_model_to_sequence(
        dict_of_sequence_actions=dict_of_sequence_actions,
        rnn_dir=rnn_dir,
        vae_dir=vae_dir,
        max_coordinate_size_for_task=448,
        device=device,
        initial_obs_path=initial_obs_path,
        load_best_rnn=load_best_rnn,
        render=False,
        temperature=temperature
    )

    comparison_loss_function = torch.nn.L1Loss()
    all_rewards = []

    all_comparison_losses = {}

    extended_log_info_as_txt = ""
    for i, (sequence_length, achieved_sequence_rewards) in enumerate(compared_rewards.items()):
        all_rewards.extend(achieved_sequence_rewards)

        actual_rewards = dict_of_sequence_rewards[sequence_length]

        comparison_loss = comparison_loss_function(
            torch.tensor(achieved_sequence_rewards), torch.tensor(actual_rewards)
        )
        all_comparison_losses[sequence_length] = comparison_loss

        extended_log_info_as_txt += (f"seq_len {sequence_length} # {len(achieved_sequence_rewards)} "
                                     f"- Actual Rew {np.mean(actual_rewards):.6f} "
                                     f"- Cmp {comparison_loss:.6f} "
                                     f"- Max {np.max(achieved_sequence_rewards):.6f} "
                                     f"- Mean {np.mean(achieved_sequence_rewards):.6f} "
                                     f"- Std {np.std(achieved_sequence_rewards):.6f} "
                                     f"- Min {np.min(achieved_sequence_rewards):.6f}  \n")

    return all_comparison_losses, all_rewards, extended_log_info_as_txt


@click.command()
@click.option("-r", "--rnn-dir", type=str, required=True, help="Path to a trained MDN RNN directory")
@click.option("-d", "--dataset", "dataset_name", type=str,
              help="Dataset name, if not used then dataset of MDN RNN is used")
@click.option("--dataset-path", type=str, help="Path of the dataset specified by '--dataset'")
@click.option("-g", "--gpu", type=int, default=-1, help="Use CPU (-1) or the corresponding GPU to load the models")
@click.option("-t", "--temperatures", type=float, default=[1.0], multiple=True,
              help="Temperature parameter(s) of MDNRNN")
@click.option("--best-rnn/--no-best-rnn", "load_best_rnn", type=bool, default=True,
              help="Load the best RNN or the last checkpoint")
@click.option("--vae-copied/--no-vae-copied", type=bool, default=True, help="Was the VAE copied?")
@click.option("--vae-location", type=str, default="local", help="Where was the vae trained (for example ai-machine)?")
def main(rnn_dir: str, dataset_name: str, dataset_path: str, gpu: int, temperatures: Tuple[float], load_best_rnn: bool,
         vae_copied: bool, vae_location: str):
    """
    TODO
        - Save result in numpy file
        - CLI parameter to generate graph maybe
    """
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    vae_dir = get_depending_model_path("rnn", rnn_dir)
    vae_dir = resolve_model_path(vae_dir, model_copied=vae_copied, location=vae_location)

    device = get_device(gpu)

    rnn_config = load_yaml_config(os.path.join(rnn_dir, "config.yaml"))
    model_name = rnn_config["model_parameters"]["name"]
    model_type = select_rnn_model(model_name)

    if dataset_name is None:
        logging.info("Using dataset_name and dataset_path from RNN config. Might not work when the server running "
                     "this script is different from the one the RNN was trained on!")
        dataset_name = rnn_config["experiment_parameters"]["dataset"]
        dataset_path = rnn_config["experiment_parameters"]["data_path"]
    else:
        assert dataset_path is not None, "If --dataset is provided, --dataset-path has to be provided too"

    vae_preprocessed_data_path = get_vae_preprocessed_data_path_name(vae_dir, dataset_name)

    reward_transformation_function = get_rnn_reward_transformation_function(
        reward_output_mode=model_type.get_reward_output_mode(),
        reward_output_activation_function=rnn_config["model_parameters"]["reward_output_activation_function"]
    )

    actions_transformation_function = get_rnn_action_transformation_function(
        max_coordinate_size_for_task=448,
        reduce_action_coordinate_space_by=rnn_config["model_parameters"]["reduce_action_coordinate_space_by"],
        action_transformation_function_type=rnn_config["model_parameters"]["action_transformation_function"]
    )

    val_dataset, _ = get_main_rnn_data_loader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split="val",
        sequence_length=1,
        batch_size=None,
        actions_transformation_function=actions_transformation_function,
        reward_transformation_function=reward_transformation_function,
        vae_preprocessed_data_path=vae_preprocessed_data_path,
        use_shifted_data=rnn_config["experiment_parameters"]["use_shifted_data"],
        shuffle=False
    )

    for temp in temperatures:
        all_comparison_losses, all_rewards, extended_log_info_as_txt = start_reward_comparison(
            rnn_dir=rnn_dir,
            vae_dir=vae_dir,
            val_dataset=val_dataset,
            model_name=model_name,
            reward_transformation_function=reward_transformation_function,
            device=device,
            temperature=temp,
            load_best_rnn=load_best_rnn
        )

        logging.info(f"\nTemperature: {temp}")
        logging.info(f"\n{extended_log_info_as_txt}")


if __name__ == "__main__":
    main()
