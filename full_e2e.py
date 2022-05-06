import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import click
import yaml

from utils.setup_utils import initialize_logger

logger, _ = initialize_logger()
logger.setLevel(logging.INFO)


@click.command()
@click.option("--gen-seq-len", "generator_sequence_length", type=int, default=5,
              help="generate ground_truth with actions sequences of this length")
@click.option("--gen-seq-no", "generator_sequence_number", type=int, default=2,
              help="generate ground_truth with total number of actions sequences")
@click.option("--gen-work-no", "generator_worker_number", type=int, default=2,
              help="generate ground_truth with this number of paralell worker process. Recommended: Twice the CPU cores")
@click.option("--gen-monkey-type", "generator_monkey_type", type=str, default="random-clicks",
              help="generate ground_truth with this type of monkey-tester")
@click.option("--dir", "base_dir", type=str, default="_e2e",
              help="root-dir for experiments")
@click.option("--config-v", "orig_config_v", type=str, default="_e2e_cfg/2_vae_config.yaml",
              help="Path to a YAML configuration containing training options for V-Model")
@click.option("--config-m", "orig_config_m", type=str,
              help="Path to a YAML configuration containing training options for M-Model")
@click.option("--config-c", "orig_config_c", type=str,
              help="Path to a YAML configuration containing training options for C-Model")
@click.option("--comet/--no-comet", type=bool, default=False,
              help="Disable logging to Comet (automatically disabled when API key is not provided in home folder)")
@click.option("--test-gpu", type=int, default=-1, help="Number of GPU to be used for test data computation, -1 is CPU")
def main(orig_config_v: str, orig_config_m: str, orig_config_c: str, comet: bool, test_gpu: int,
         generator_sequence_length: int, generator_sequence_number: int, generator_worker_number: int,
         generator_monkey_type: str, base_dir: str):
    main_args = locals()
    for key in main_args:
        logging.info(f"{key}: {main_args[key]}")
    work_env = os.environ.copy()
    work_env['PYSIDE_DESIGNER_PLUGINS'] = '.'  # https://stackoverflow.com/a/68605424/5132456
    logging.info(f"work_dir: {base_dir}")
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    work_dir_generator = Path(base_dir).joinpath("01_generator")
    generate(work_dir_generator, work_env, generator_sequence_number, generator_worker_number,
             generator_sequence_length)
    work_config_v = os.path.join(base_dir, "config_v.yaml")
    shutil.copyfile(orig_config_v, work_config_v)

    replace_in_yaml(work_config_v, work_config_v,
                    key_path=['experiment_parameters'],
                    value={"aa": 123, "bb": 324})

    replace_in_yaml(work_config_v, work_config_v,
                    key_path=['logging_parameters', 'save_dir'],
                    value='_e2e_/01_generator/random-clicks/2022-05-01_12-23-34-mixed-deduplicated-images-splits')


def generate(work_dir, work_env, generator_sequence_number, generator_worker_number, generator_sequence_length):
    if work_dir.is_dir():
        logging.warning(f"dir for datageneration already exists... Skipping data generation")
        return
    if work_dir.is_file():
        raise RuntimeError(f"workdir {work_dir}is a file")

    args = [
        "python", "data/parallel_data_generation.py",
        "-s", str(generator_sequence_number),
        "-p", str(generator_worker_number),
        "--amount", str(generator_sequence_length),
        '--monkey-type=random-clicks',
        "--no-log",
        "--root-dir", str(work_dir)]
    run_script(args, work_env)


def _replace_in_dict(dictionary: dict, key_path: List[str], value: any):
    if len(key_path) == 0:
        raise RuntimeError("unexpected state")
    if len(key_path) == 1:
        key = key_path[0]
        if key in dictionary:
            logging.info(f"old value: {dictionary[key]}")
        else:
            logging.info(f"adding new key")
        dictionary[key] = value
        return dictionary
    if len(key_path) > 1:
        next_key = key_path.pop(0)
        if isinstance(dictionary[next_key], dict):
            return _replace_in_dict(dictionary[next_key], key_path, value)
        logging.error("expected dict instead found " + dictionary[next_key])
        return dictionary


def replace_in_yaml(input_yaml: str, output_yaml: str, key_path: List[str], value: any):
    with open(input_yaml) as f:
        yaml_content = yaml.safe_load(f)

    logging.info("replacing key " + ".".join(key_path) + " with value " + str(value))
    _replace_in_dict(yaml_content, key_path, value)

    with open(output_yaml, "w") as f:
        yaml.dump(yaml_content, f)


def run_script(args: List[str], work_env: dict):
    logging.info('running script: ' + ' '.join(args))
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True,
                            env=work_env)
    for line in proc.stdout:
        # filter annoying warnings and clutter
        if 'qt.pysideplugin' in line \
                or 'Qt WebEngine seems to be initialized from a plugin' in line:
            # warnings for data-generator
            continue
        print(line, end="")
    logging.info('script done')


#
#


if __name__ == "__main__":
    main()
