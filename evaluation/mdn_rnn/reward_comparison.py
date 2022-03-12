from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

from envs.simulated_gui_env import SimulatedGUIEnv


def compare_reward_of_m_model_to_sequence(
        dict_of_sequence_actions: Dict[int, List[List[Tuple[torch.Tensor, torch.Tensor]]]], rnn_dir: str,
        vae_dir: str, max_coordinate_size_for_task: int, device: torch.device, initial_obs_path: str,
        load_best_rnn: bool = True, render: bool = False
) -> Dict[int, List[float]]:

    env = SimulatedGUIEnv(
        rnn_dir=rnn_dir,
        vae_dir=vae_dir,
        initial_obs_path=initial_obs_path,
        max_coordinate_size_for_task=max_coordinate_size_for_task,
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
