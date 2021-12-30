import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Tuple

import click
import gym
# noinspection PyUnresolvedReferences
import gym_gui_environments
import numpy as np
from PIL import Image

from utils.misc import initialize_logger

RANDOM_CLICK_MONKEY_TYPE = "random-clicks"
RANDOM_WIDGET_MONKEY_TYPE = "random-widgets"


def _save_observation(observation: np.ndarray, iteration: int, observations_directory: str):
    im = Image.fromarray(observation)
    # Add leading zeros to the filename so that they are properly sorted
    file_name = f"{iteration}".zfill(8)
    im.save(os.path.join(observations_directory, f"{file_name}.png"))


def _store_data(data: dict, iteration: int, action: Tuple[int, int], reward: float):
    data[iteration] = {"action": action, "reward": reward}

    return data


def _rollout_one_iteration(env, current_iteration: int, observations_directory: str,
                           reward_sum: float, rewards: list, actions: list) -> Tuple[float, list, list]:
    reward, observation, done, info = env.step(True)
    _save_observation(observation, current_iteration, observations_directory)

    reward_sum += reward / 100.0

    if current_iteration % 500 == 0:
        logging.info(
            f"{current_iteration}: Current reward '{reward_sum}'"
        )

    rewards += [reward]
    actions += [[info["x"], info["y"]]]

    return reward_sum, rewards, actions


def _time_mode_rollout(amount: int, env, observations_directory: str) -> Tuple[float, list, list]:
    i = 1
    reward_sum = 0
    rewards = []
    actions = []

    start_time = time.time()

    while time.time() < start_time + amount:
        reward_sum, rewards, actions = _rollout_one_iteration(env, i, observations_directory, reward_sum, rewards,
                                                              actions)
        i += 1

    return reward_sum, rewards, actions


def _iteration_mode_rollout(amount: int, env, observations_directory: str) -> Tuple[float, list, list]:
    reward_sum = 0
    rewards = []
    actions = []

    for i in range(1, amount + 1):
        reward_sum, rewards, actions = _rollout_one_iteration(env, i, observations_directory, reward_sum, rewards,
                                                              actions)

    return reward_sum, rewards, actions


def start_monkey_tester(env: gym.Env, stop_mode: str, amount: int, chosen_directory: str, observations_directory: str):
    observation = env.reset()
    _save_observation(observation, iteration=0, observations_directory=observations_directory)

    if stop_mode == "time":
        reward_sum, rewards, actions = _time_mode_rollout(amount, env, observations_directory)
    else:
        reward_sum, rewards, actions = _iteration_mode_rollout(amount, env, observations_directory)

    np.savez(
        os.path.join(chosen_directory, "data.npz"),
        rewards=np.array(rewards, dtype=np.float32),
        actions=np.array(actions, dtype=np.int32)
    )

    logging.info(f"Finished data generation with a summed up reward of {reward_sum}")
    env.close()


@click.command()
@click.option("-t", "--time", "stop_mode", flag_value="time", default=True,
              help="Use elapsed time in seconds to stop the data generation")
@click.option("-i", "--iterations", "stop_mode", flag_value="iterations",
              help="Use the number of iterations to stop the data generation")
@click.option("--amount", type=int,
              help="Amount on how long the data generation shall run (seconds or number of iterations, depending on "
                   "the stop_mode")
@click.option("-m", "--monkey-type", default=RANDOM_WIDGET_MONKEY_TYPE,
              type=click.Choice([RANDOM_CLICK_MONKEY_TYPE, RANDOM_WIDGET_MONKEY_TYPE]),
              show_default=True, help="Choose which type of random monkey tester to use")
@click.option("--root-dir", type=str, default="datasets/gui_env",
              help="In this directory, subfolders are automatically created based on time to store the generated data")
@click.option("--directory", type=str,
              help="This directory is directly used to store the data, instead of automatically generated subfolders "
                   "as with --root-dir")
@click.option("--random-click-prob", type=float,
              help="If the random widget monkey tester is chosen, use this to define the probability for random clicks")
@click.option("--log/--no-log", type=bool, default=True, help="If true set logging level to debug and log to a file")
@click.option("--html-report/--no-html-report", type=bool, default=True,
              help="If true, save the HTML Report of the coverage")
def main(stop_mode: str, amount: int, monkey_type: str, root_dir: str, directory: str, random_click_prob: float,
         log: bool, html_report: bool):
    if directory is not None:
        chosen_directory = directory
    else:
        chosen_directory = os.path.join(root_dir, monkey_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        shutil.rmtree(chosen_directory)
    except FileNotFoundError:
        pass

    os.makedirs(chosen_directory, exist_ok=True)

    observations_directory = os.path.join(chosen_directory, "observations")
    os.makedirs(observations_directory, exist_ok=True)

    logger, formatter = initialize_logger()

    if log:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(chosen_directory, "monkey_tester.log"), "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger.setLevel(logging.INFO)

    html_report_directory = None

    if html_report:
        html_report_directory = os.path.join(chosen_directory, "html-report")

    env_kwargs = {"generate_html_report": html_report, "html_report_directory": html_report_directory}

    if monkey_type == RANDOM_CLICK_MONKEY_TYPE:
        env_id = "PySideGUIRandomClick-v0"
    else:
        env_id = "PySideGUIRandomWidget-v0"

        if random_click_prob is not None:
            env_kwargs["random_click_probability"] = random_click_prob

    env = gym.make(env_id, **env_kwargs)

    start_monkey_tester(env, stop_mode, amount, chosen_directory, observations_directory)

    chosen_options = {
        "env-used": env_id,
        "stop-mode": stop_mode,
        "amount": amount,
        "monkey-type": monkey_type,
        "root-dir": root_dir,
        "explicit-dir": directory,
        "random-click-probability": random_click_prob,
        "log": log,
        "html-report": html_report
    }

    with open(os.path.join(chosen_directory, "data_generation_options.json"), "w", encoding="utf-8") as f:
        json.dump(chosen_options, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
