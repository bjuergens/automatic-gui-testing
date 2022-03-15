from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from envs.simulated_gui_env import SimulatedGUIEnv
from utils.training_utils.training_utils import generate_initial_observation_latent_vector


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


def start_reward_comparison(rnn_dir, vae_dir, val_dataset, model_name, reward_transformation_function, device):
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
        load_best_rnn=True,
        render=False,
        temperature=1.0
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
