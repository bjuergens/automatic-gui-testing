import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import click
import numpy as np
from PIL import Image

from envs.gui_env.gui_env import GUIEnvRandomWidget, GUIEnvRandomClick

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


def _rollout_one_iteration(env, current_iteration: int, data: dict, observations_directory: str,
                           reward_sum: float) -> Tuple[dict, float]:
    reward, observation, done, info = env.step()
    _save_observation(observation, current_iteration - 1, observations_directory)

    reward_sum += reward

    if current_iteration % 500 == 0:
        logging.info(
            f"{current_iteration}: Current reward '{reward_sum}'"
        )

    data = _store_data(data, current_iteration - 1, action=(info["x"], info["y"]), reward=reward)

    return data, reward_sum


def _time_mode_rollout(amount: int, env, observations_directory: str) -> Tuple[dict, float]:
    i = 1
    reward_sum = 0
    data = {}
    start_time = time.time()

    while time.time() < start_time + amount:
        data, reward_sum = _rollout_one_iteration(env, i, data, observations_directory, reward_sum)
        i += 1

    return data, reward_sum


def _iteration_mode_rollout(amount: int, env, observations_directory: str) -> Tuple[dict, float]:
    reward_sum = 0
    data = {}

    for i in range(1, amount + 1):
        data, reward_sum = _rollout_one_iteration(env, i, data, observations_directory, reward_sum)

    return data, reward_sum


def start_monkey_tester(stop_mode: str, amount: int, monkey_type: str, chosen_directory: str,
                        observations_directory: str, random_click_prob: Optional[float], html_report: bool,
                        html_report_directory: Optional[str]):

    if monkey_type == RANDOM_CLICK_MONKEY_TYPE:
        env = GUIEnvRandomClick(html_report, html_report_directory)
    else:
        if random_click_prob is not None:
            env = GUIEnvRandomWidget(random_click_prob, generate_html_report=html_report,
                                     html_report_directory=html_report_directory)
        else:
            # Use the default value for the random click probability
            env = GUIEnvRandomWidget(generate_html_report=html_report, html_report_directory=html_report_directory)

    observation = env.reset()
    _save_observation(observation, iteration=0, observations_directory=observations_directory)

    if stop_mode == "time":
        data, reward_sum = _time_mode_rollout(amount, env, observations_directory)
    else:
        data, reward_sum = _iteration_mode_rollout(amount, env, observations_directory)

    with open(os.path.join(chosen_directory, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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

    logger = logging.getLogger("")
    formatter = logging.Formatter('[%(asctime)s] - %(funcName)s - %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

    if log:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(chosen_directory, "monkey_tester.log"), "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    html_report_directory = None

    if html_report:
        html_report_directory = os.path.join(chosen_directory, "html-report")

    start_monkey_tester(stop_mode, amount, monkey_type, chosen_directory, observations_directory, random_click_prob,
                        html_report, html_report_directory)


if __name__ == "__main__":
    main()