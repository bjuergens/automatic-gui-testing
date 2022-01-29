import logging
import os
import random
import subprocess
import tempfile
from copy import deepcopy

from utils.setup_utils import save_yaml_config, initialize_logger


def start_vae_training(config: dict):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        config_file_path = os.path.join(tmp_dir_name, "config.yaml")
        save_yaml_config(config_file_path, config)

        logging.info("Starting new experiment")
        subprocess.call(["python", "train_vae.py", "--config", config_file_path])
        logging.info("Finished experiment")


def select_hyperparameters(config: dict):
    new_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            new_config[k] = select_hyperparameters(v)
        elif isinstance(v, list):
            new_config[k] = random.choice(v)
        else:
            new_config[k] = v
    return new_config


def main():
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    original_config = {
        'model_parameters': {
            'name': 'small_filter_sizes_maxpool',
            'activation_function': 'relu',
            'hidden_dimensions': [[2, 4, 4, 4, 4, 8]],
            'latent_size': 16,
            'input_channels': 3,
            'batch_norm': True,
            'kld_weight': 1.0,
            'kld_warmup': True},
        'experiment_parameters': {
            'dataset': 'single_sequence_vae',
            'dataset_path': 'datasets/gui_env/random-widgets/2021-12-29_19-02-29-mixed',
            'img_size': 448,
            'batch_size': 2,
            'learning_rate': 0.0005,
            'kld_weight': 1.0,
            'kld_warmup': False,
            'max_epochs': 1,
            'manual_seed': 1010},
        'trainer_parameters': {
            'gpu': -1,
            'num_workers': 0},
        'logging_parameters': {
            'debug': False,
            'scalar_log_frequency': 20,
            'image_epoch_log_frequency': 5,
            'save_dir': 'logs/vae/'}
    }

    chosen_parameters = select_hyperparameters(original_config)

    new_config = deepcopy(chosen_parameters)
    new_config["model_parameters"]["latent_size"] = 32

    start_vae_training(new_config)

    logging.info("Finished scheduled training!")


if __name__ == "__main__":
    main()
