import logging
import os

import click
import torch
from test_tube import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_implementations import get_rnn_dataloader
from models import select_rnn_model
from models.rnn import BaseRNN
from utils.data_processing_utils import preprocess_observations_with_vae
from utils.setup_utils import initialize_logger, load_yaml_config, set_seeds, get_device, save_yaml_config
from utils.training_utils import load_vae_architecture, save_checkpoint, vae_transformation_functions
from utils.training_utils.average_meter import AverageMeter

# from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
# from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
# from utils.learning import ReduceLROnPlateau


def data_pass(model: BaseRNN, vae, experiment, optimizer, data_loader: DataLoader, device: torch.device,
              current_epoch: int, global_log_step: int, train: bool):
    if train:
        model.train()
        loss_key = "loss"
        latent_loss_key = "latent_loss"
        reward_loss_key = "reward_loss"
    else:
        model.eval()
        loss_key = "val_loss"
        latent_loss_key = "val_latent_loss"
        reward_loss_key = "val_reward_loss"

    total_loss_meter = AverageMeter(loss_key, ":.4f")
    latent_loss_meter = AverageMeter(latent_loss_key, ":.4f")
    reward_loss_meter = AverageMeter(reward_loss_key, ":.4f")

    old_dataset_index = None

    log_interval = 20

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), unit="batch", desc=f"Epoch {current_epoch}")

    for i, data in progress_bar:
        mus, next_mus, log_vars, next_log_vars, rewards, actions = [d.to(device) for d in data[0]]
        dataset_indices: torch.Tensor = data[1]
        current_dataset_index = dataset_indices[0]
        assert all(dataset_indices == current_dataset_index)

        if old_dataset_index is None or old_dataset_index != current_dataset_index:
            old_dataset_index = current_dataset_index
            model.initialize_hidden()

        batch_size = mus.size(0)
        latent_obs = vae.reparameterize(mus, log_vars)
        latent_next_obs = vae.reparameterize(next_mus, next_log_vars)

        if train:
            optimizer.zero_grad()
            model_output = model(latent_obs, actions)
            loss, (latent_loss, reward_loss) = model.loss_function(next_latent_vector=latent_next_obs, reward=rewards,
                                                                   model_output=model_output)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model_output = model(latent_obs, actions)
                loss, (latent_loss, reward_loss) = model.loss_function(next_latent_vector=latent_next_obs,
                                                                       reward=rewards, model_output=model_output)

        total_loss_meter.update(loss.item(), batch_size)
        latent_loss_meter.update(latent_loss, batch_size)
        reward_loss_meter.update(reward_loss, batch_size)

        if i % log_interval == 0 or i == (len(data_loader) - 1):
            progress_bar.set_postfix_str(f"loss={total_loss_meter.avg:.4f} latent={latent_loss_meter.avg:.4f} "
                                         f"reward={reward_loss_meter.avg:.4f}")

            experiment.log({
                loss_key: total_loss_meter.avg,
                latent_loss_key: latent_loss_meter.avg,
                reward_loss_key: reward_loss_meter.avg
            }, global_step=global_log_step)

            global_log_step += 1

    progress_bar.close()

    experiment.log({
        f"epoch_{loss_key}": total_loss_meter.avg
    }, global_step=current_epoch)

    return total_loss_meter.avg, global_log_step


@click.command()
@click.option("-c", "--config", "config_path", type=str, required=True,
              help="Path to a YAML configuration containing training options")
def main(config_path: str):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    config = load_yaml_config(config_path)

    batch_size = config["experiment_parameters"]["batch_size"]
    sequence_length = config["experiment_parameters"]["sequence_length"]
    learning_rate = config["experiment_parameters"]["learning_rate"]
    max_epochs = config["experiment_parameters"]["max_epochs"]

    dataset_name = config["experiment_parameters"]["dataset"]
    dataset_path = config["experiment_parameters"]["data_path"]

    num_workers = config["trainer_parameters"]["num_workers"]
    gpu_id = config["trainer_parameters"]["gpu"]

    manual_seed = config["experiment_parameters"]["manual_seed"]

    set_seeds(manual_seed)
    device = get_device(gpu_id)

    base_save_dir = config["logging_parameters"]["base_save_dir"]
    model_name = config["model_parameters"]["name"]

    vae_directory = config["vae_parameters"]["directory"]
    vae, vae_name = load_vae_architecture(vae_directory, device, load_best=True)

    vae_config = load_yaml_config(os.path.join(vae_directory, "config.yaml"))
    latent_size = vae_config["model_parameters"]["latent_size"]
    img_size = vae_config["experiment_parameters"]["img_size"]

    model_type = select_rnn_model(model_name)
    model = model_type(config["model_parameters"], latent_size, batch_size, device).to(device)

    # optimizer = torch.optim.RMSprop(mdn_rnn.parameters(), lr=learning_rate, alpha=.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # earlystopping = EarlyStopping('min', patience=30)

    # if exists(rnn_file) and not args.noreload:
    #     rnn_state = torch.load(rnn_file)
    #     print("Loading MDRNN at epoch {} "
    #           "with test error {}".format(
    #         rnn_state["epoch"], rnn_state["precision"]))
    #     mdrnn.load_state_dict(rnn_state["state_dict"])
    #     optimizer.load_state_dict(rnn_state["optimizer"])
    #     scheduler.load_state_dict(state['scheduler'])
    #     earlystopping.load_state_dict(state['earlystopping'])

    transformation_functions = vae_transformation_functions(img_size)

    vae_output_file_name = preprocess_observations_with_vae(dataset_path, vae, vae_name=vae_name,
                                                            vae_version=vae_directory.split("version_")[-1],
                                                            img_size=img_size, device=device, force=False)

    additional_dataloader_kwargs = {"num_workers": num_workers, "pin_memory": True}

    train_dataloader = get_rnn_dataloader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split="train",
        vae_output_file_name=vae_output_file_name,
        sequence_length=sequence_length,
        batch_size=batch_size,
        **additional_dataloader_kwargs
    )

    val_dataloader = get_rnn_dataloader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split="val",
        vae_output_file_name=vae_output_file_name,
        sequence_length=sequence_length,
        batch_size=batch_size,
        **additional_dataloader_kwargs
    )

    save_dir = os.path.join(base_save_dir, dataset_name)
    global_train_log_steps = 0
    global_val_log_steps = 0

    experiment = Experiment(
        save_dir=save_dir,
        name=model_name,
        debug=config["logging_parameters"]["debug"],  # Turns off logging if True
        create_git_tag=False,
        autosave=True
    )

    experiment.tag(config)

    if not experiment.debug:
        log_dir = experiment.get_logdir().split("tf")[0]
        best_model_filename = os.path.join(log_dir, "best.pt")
        checkpoint_filename = os.path.join(log_dir, "checkpoint.pt")

        save_yaml_config(os.path.join(log_dir, "config.yaml"), config)
    else:
        # Enables debugging of the gradient calculation, shows where errors/NaN etc. occur
        torch.autograd.set_detect_anomaly(True)

    training_version = experiment.version
    if training_version is not None:
        logging.info(f"Started MDN-RNN training version_{training_version} for {max_epochs} epochs")

    current_best = None
    for current_epoch in range(max_epochs):
        _, global_train_log_steps = data_pass(model, vae, experiment, optimizer, train_dataloader, device,
                                              current_epoch, global_train_log_steps, train=True)

        val_loss, global_val_log_steps = data_pass(model, vae, experiment, optimizer, val_dataloader, device,
                                                   current_epoch, global_val_log_steps, train=False)

        # scheduler.step(test_loss)
        # earlystopping.step(test_loss)
        if not experiment.debug:
            is_best = not current_best or val_loss < current_best

            if is_best:
                current_best = val_loss

            save_checkpoint({
                "epoch": current_epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                # 'earlystopping': earlystopping.state_dict(),
                # "precision": test_loss,
            }, is_best=is_best, checkpoint_filename=checkpoint_filename, best_filename=best_model_filename)

        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(e))
        #     break

    if not experiment.debug:
        # Ensure everything is logged to the tensorboard
        experiment.flush()


if __name__ == "__main__":
    main()
