import contextlib
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
@click.option("--gen-seq-len", "generator_sequence_length", type=int, default=50,
              help="generate ground_truth with actions sequences of this length")
@click.option("--gen-seq-no", "generator_sequence_number", type=int, default=128,
              help="generate ground_truth with total number of actions sequences")
@click.option("--gen-work-no", "generator_worker_number", type=int, default=32,
              help="generate ground_truth with this number of parallel worker process. Recommended: Twice the CPU cores")
@click.option("--gen-monkey-type", "generator_monkey_type", type=str, default="random-clicks",
              help="generate ground_truth with this type of monkey-tester")
@click.option("--dir", "base_dir", type=str, default="_e2e",
              help="root-dir for experiments")
@click.option("--config-v", "orig_config_v", type=str, default="configs/e2e_cfg/2_vae_config.yaml",
              help="Path to a YAML configuration containing training options for V-Model")
@click.option("--config-m", "orig_config_m", type=str, default="e2e_cfg/e2e_cfg/3_mdn_rnn_config.yaml",
              help="Path to a YAML configuration containing training options for M-Model")
@click.option("--config-c", "orig_config_c", type=str, default="e2e_cfg/e2e_cfg/4_controller_config.yaml",
              help="Path to a YAML configuration containing training options for C-Model")
@click.option("--comet/--no-comet", type=bool, default=False,
              help="Disable logging to Comet (automatically disabled when API key is not provided in home folder)")
@click.option("--test-gpu", type=int, default=-1, help="Number of GPU to be used for test data computation, -1 is CPU")
def main(orig_config_v: str, orig_config_m: str, orig_config_c: str, comet: bool, test_gpu: int,
         generator_sequence_length: int, generator_sequence_number: int, generator_worker_number: int,
         generator_monkey_type: str, base_dir: str):
    full_run = True  # helper var for debugging
    main_args = locals()
    for key in main_args:
        logging.info(f"{key}: {main_args[key]}")
    work_env = os.environ.copy()
    work_env['PYSIDE_DESIGNER_PLUGINS'] = '.'  # https://stackoverflow.com/a/68605424/5132456
    logging.info(f"work_dir: {base_dir}")

    work_dir_generator = Path(base_dir).joinpath("01_generator")
    work_dir_vae = Path(base_dir).joinpath("02_vae")
    work_dir_rnn = Path(base_dir).joinpath("03_rnn")
    work_dir_ctl = Path(base_dir).joinpath("04_ctl")

    vae_data_path = os.path.join(work_dir_vae, "data")
    rnn_data_path = os.path.join(work_dir_rnn, "data")

    log_dir_vae = Path(base_dir) / "log" / "vae"
    log_dir_rnn = Path(base_dir) / "log" / "rnn"
    log_dir_ctl = Path(base_dir) / "log" / "ctl"
    log_dir_vae.mkdir(parents=True, exist_ok=True)
    log_dir_rnn.mkdir(parents=True, exist_ok=True)
    log_dir_ctl.mkdir(parents=True, exist_ok=True)
    Path(work_dir_vae).mkdir(parents=True, exist_ok=True)
    Path(work_dir_rnn).mkdir(parents=True, exist_ok=True)
    Path(work_dir_ctl).mkdir(parents=True, exist_ok=True)

    if full_run:
        generate(work_dir=work_dir_generator,
                 work_env=work_env,
                 generator_sequence_number=generator_sequence_number,
                 generator_worker_number=generator_worker_number,
                 generator_sequence_length=generator_sequence_length)

        prepare_vae_data(work_dir_generator, work_env, vae_data_path)
        train_vae(work_dir_vae, work_env, orig_config_v, vae_data_path, log_dir_vae)

    prepare_rnn_data(work_dir_generator, vae_data_path, work_env, rnn_data_path, generator_sequence_length)
    train_rnn(work_dir_rnn,
              work_env=work_env,
              orig_config=orig_config_m,
              rnn_data_path=rnn_data_path,
              log_dir_vae=log_dir_vae,
              log_dir_rnn=log_dir_rnn,
              rnn_sequence_length=generator_sequence_length // 2)

    train_controller(work_dir_ctl, work_env, orig_config_c, rnn_data_path, log_dir_vae, log_dir_ctl, log_dir_rnn)


def _get_most_recent_log(data_set_path_log_path):
    versions = os.listdir(data_set_path_log_path)
    versions.sort(reverse=True)
    most_recent = str(versions[0])
    most_recent_path = Path(data_set_path_log_path) / most_recent
    return most_recent_path


def train_controller(work_dir, work_env, orig_config, rnn_data_path, log_dir_vae, log_dir, log_dir_rnn):

    work_config = os.path.join(work_dir, "config_c.yaml")
    with contextlib.suppress(FileNotFoundError):
        os.remove(work_config)
    shutil.copyfile(orig_config, work_config)

    most_recent_path = _get_most_recent_log(Path(log_dir_rnn)
                                            / "multiple_sequences_varying_length_individual_data_loaders_rnn")

    replace_in_yaml(work_config, work_config,
                    key_path=['rnn_parameters', 'rnn_dir'],
                    value=str(most_recent_path))

    replace_in_yaml(work_config, work_config,
                    key_path=['logging_parameters', 'save_dir'],
                    value=str(log_dir))

    args_c_train = ["python", "train_controller.py",
                    "-c", work_config,
                    "--disable-comet"]
    run_script(args_c_train, work_env)


def prepare_rnn_data(work_dir_generator, vae_data_path, work_env, target_path, sequence_length, skip_of_exists=False):
    dirpath = Path(target_path)
    if dirpath.exists() and dirpath.is_dir():
        if skip_of_exists:
            logging.info(f"skipping because target dir {dirpath} already exists ")
            return
        logging.info(f"cleaning up old datapath {dirpath}")
        shutil.rmtree(dirpath)

    target_train = Path(target_path) / "train" / str(sequence_length)
    target_val = Path(target_path) / "val" / str(sequence_length)
    target_test = Path(target_path) / "test" / str(sequence_length)

    Path(target_train).parent.mkdir(parents=True, exist_ok=True)
    Path(target_val).parent.mkdir(parents=True, exist_ok=True)
    Path(target_test).parent.mkdir(parents=True, exist_ok=True)

    for generator_type in os.listdir(work_dir_generator):
        gen_type_path = os.path.join(work_dir_generator, generator_type)
        for experiment_date in os.listdir(gen_type_path):
            exp_path = Path(gen_type_path) / experiment_date
            logging.info(f"found generator_run in {exp_path}")
            logging.info(f"use everything for training at {target_train}")
            shutil.copytree(exp_path, target_train)

            logging.info(f"use first for val at {target_val}")
            shutil.copytree(exp_path / "0", target_val / "0")

            logging.info(f"use second for test at {target_test}")
            shutil.copytree(exp_path / "1", target_test / "1")


def train_rnn(work_dir, work_env, orig_config, rnn_data_path, log_dir_vae, log_dir_rnn, rnn_sequence_length):
    work_config = os.path.join(work_dir, "config_m.yaml")
    with contextlib.suppress(FileNotFoundError):
        os.remove(work_config)
    shutil.copyfile(orig_config, work_config)

    most_recent_path = _get_most_recent_log(Path(log_dir_vae) / "gui_env_image_dataset")

    replace_in_yaml(work_config, work_config,
                    key_path=['vae_parameters', 'directory'],
                    value=str(most_recent_path))
    replace_in_yaml(work_config, work_config,
                    key_path=['experiment_parameters', 'sequence_length'],
                    value=rnn_sequence_length)
    replace_in_yaml(work_config, work_config,
                    key_path=['experiment_parameters', 'data_path'],
                    value=str(rnn_data_path))
    replace_in_yaml(work_config, work_config,
                    key_path=['logging_parameters', 'base_save_dir'],
                    value=str(log_dir_rnn))
    args_v_train = ["python", "train_mdn_rnn.py",
                    "-c", work_config,
                    "--disable-comet"]
    run_script(args_v_train, work_env)


def prepare_vae_data(work_dir_generator, work_env, target_path, skip_of_exists=True):
    dirpath = Path(target_path)
    if dirpath.exists() and dirpath.is_dir():
        if skip_of_exists:
            logging.info(f"skipping because target dir {dirpath} already exists ")
            return
        logging.info(f"cleaning up old datapath {dirpath}")
        shutil.rmtree(dirpath)
    for generator_type in os.listdir(work_dir_generator):
        gen_type_path = os.path.join(work_dir_generator, generator_type)
        for experiment_date in os.listdir(gen_type_path):
            exp_path = os.path.join(gen_type_path, experiment_date)
            mixed_path = exp_path + "-mixed"
            dedup_path = mixed_path + "-deduplicated-images"
            split_path = dedup_path + "-splits"

            logging.info(f"found generator_run in {exp_path}")

            args_copy = ["python", "data/data_processing/copy_images.py",
                         "-d", exp_path]
            run_script(args_copy, work_env)
            logging.info(f"total files in mixed: {len(os.listdir(mixed_path))}")

            args_dedup = ["python", "data/data_processing/remove_duplicate_images.py",
                          "-d", mixed_path]
            run_script(args_dedup, work_env)
            dedup_file_count = len(os.listdir(dedup_path))
            if dedup_file_count < 10:
                logging.warning("too few files left after deduplication. Expect problems in later steps.")
            logging.info(f"total files in dedup: {dedup_file_count}")
            args_dedup = ["python", "data/data_processing/create_dataset_splits.py",
                          "-d", dedup_path]
            run_script(args_dedup, work_env, raise_exceptions=False)
            logging.info(f"total files in split: {len(os.listdir(split_path))}")

            shutil.move(split_path, target_path)

            logging.info(f"cleanup intermediate files: {mixed_path}")
            shutil.rmtree(mixed_path)
            logging.info(f"cleanup intermediate files: {dedup_path}")
            shutil.rmtree(dedup_path)


def train_vae(work_dir, work_env, orig_config_v, vae_data_path: Path, log_dir: Path):
    work_config_v = os.path.join(work_dir, "config_v.yaml")
    shutil.copyfile(orig_config_v, work_config_v)
    replace_in_yaml(work_config_v, work_config_v,
                    key_path=['experiment_parameters', 'dataset_path'],
                    value=str(vae_data_path))
    replace_in_yaml(work_config_v, work_config_v,
                    key_path=['logging_parameters', 'save_dir'],
                    value=str(log_dir))
    args_v_train = ["python", "train_vae.py",
                    "-c", work_config_v,
                    "--disable-comet"]
    run_script(args_v_train, work_env)


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
        "-i",
        "--amount", str(generator_sequence_length),
        '--monkey-type=random-clicks',
        "--no-log",
        "--no-html-report",
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


def run_script(args: List[str], work_env: dict, raise_exceptions=True):
    logging.info('running script: ' + ' '.join(args))
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True,
                            env=work_env)
    assume_error = False
    for line in proc.stdout:
        # filter annoying warnings and clutter
        if 'qt.pysideplugin' in line \
                or 'Qt WebEngine seems to be initialized from a plugin' in line:
            # warnings for data-generator
            continue
        if 'Traceback' in line:
            assume_error = True
        print(line, end="")
    logging.info('script done. returncode: ' + str(proc.returncode))
    if assume_error:
        if raise_exceptions:
            raise RuntimeError("subprocess probably had an error")
        else:
            logging.warning("subprocess probably had an error")


#
#


if __name__ == "__main__":
    main()
