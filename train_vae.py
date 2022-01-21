import logging
import os

import click
import torch
import torch.utils.data
from test_tube import Experiment
from torch import optim
from tqdm import tqdm

from data.dataset_implementations import get_vae_dataloader
from models import select_vae_model
from utils.setup_utils import initialize_logger, load_yaml_config, set_seeds, get_device, save_yaml_config
from utils.training_utils import save_checkpoint, vae_transformation_functions, load_vae_architecture


# from utils.misc import LSIZE, RED_SIZE
# WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
# from utils.learning import EarlyStopping
# from utils.learning import ReduceLROnPlateau
# from data.loaders import RolloutObservationDataset


def train(model, experiment, train_loader, optimizer, device, current_epoch, max_epochs):
    """ One training epoch """
    model.train()
    # dataset_train.load_next_buffer()
    train_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), unit="batch",
                        desc=f"Epoch {current_epoch} - Train")

    log_interval = 20

    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss, mse_loss, kld_loss = model.loss_function(data, recon_batch, mu, log_var, current_epoch, max_epochs)
        loss.backward()
        train_loss += loss.item() * data.size(0)
        optimizer.step()

        # if batch_idx % 20 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         current_epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

        loss_float = loss.item()
        progress_bar.set_postfix_str(
            f"loss={loss_float:.4f} mse={mse_loss:.4f} kld={kld_loss:.4f}"
        )

        if batch_idx % log_interval == 0:
            experiment.log({
                "loss": loss_float,
                "reconstruction_loss": mse_loss,
                "kld": kld_loss
            })

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #     current_epoch, train_loss / len(train_loader.dataset)))

    progress_bar.close()

    experiment.log({
        "epoch_train_loss": train_loss / len(train_loader.dataset)
    })


def validate(model, experiment: Experiment, val_loader, device, current_epoch, max_epochs):
    """ One test epoch """
    model.eval()

    val_loss_sum = 0

    logged_one_batch = False

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), unit="batch",
                        desc=f"Epoch {current_epoch} - Validation")

    log_interval = 20

    for batch_idx, data in progress_bar:
        data = data.to(device)

        with torch.no_grad():
            recon_batch, mu, log_var = model(data)
            val_loss, val_mse_loss, val_kld_loss = model.loss_function(
                data, recon_batch, mu, log_var, current_epoch, max_epochs
            )

        val_loss_float = val_loss.item()

        progress_bar.set_postfix_str(
            f"val_loss={val_loss_float:.4f} val_mse={val_mse_loss:.4f} val_kld={val_kld_loss:.4f}"
        )

        val_loss_sum += val_loss_float * data.size(0)

        if not logged_one_batch and not experiment.debug:
            experiment.add_images("originals", data, global_step=current_epoch)
            experiment.add_images("reconstructions", recon_batch, global_step=current_epoch)
            logged_one_batch = True

        if batch_idx % log_interval == 0:
            experiment.log({
                "val_loss": val_loss_float,
                "val_reconstruction_loss": val_mse_loss,
                "val_kld": val_kld_loss
            })

    progress_bar.close()

    val_loss_sum /= len(val_loader.dataset)

    experiment.log({
        "epoch_val_loss": val_loss_sum
    })

    return val_loss_sum


@click.command()
@click.option("-c", "--config", "config_path", type=str, required=True,
              help="Path to a YAML configuration containing training options")
@click.option("-l", "--load", "load_path", type=str,
              help=("Path to a previous training, from which training shall continue (will create a new experiment "
                    "directory)"))
def main(config_path: str, load_path: str):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    config = load_yaml_config(config_path)

    use_kld_warmup = config["experiment_parameters"]["kld_warmup"]
    kld_weight = config["experiment_parameters"]["kld_weight"]
    batch_size = config["experiment_parameters"]["batch_size"]
    manual_seed = config["experiment_parameters"]["manual_seed"]

    dataset_name = config["experiment_parameters"]["dataset"]
    dataset_path = config["experiment_parameters"]["dataset_path"]

    img_size = config["experiment_parameters"]["img_size"]

    number_of_workers = config["trainer_parameters"]["num_workers"]
    gpu_id = config["trainer_parameters"]["gpu"]

    # VAE configuration
    vae_name = config["model_parameters"]["name"]

    set_seeds(manual_seed)
    device = get_device(gpu_id)

    transformation_functions = vae_transformation_functions(img_size)

    additional_dataloader_kwargs = {"num_workers": number_of_workers, "pin_memory": True}

    train_loader = get_vae_dataloader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split="train",
        transformation_functions=transformation_functions,
        batch_size=batch_size,
        shuffle=True,
        **additional_dataloader_kwargs
    )

    val_loader = get_vae_dataloader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split="val",
        transformation_functions=transformation_functions,
        batch_size=batch_size,
        shuffle=False,
        **additional_dataloader_kwargs
    )

    if load_path is not None:
        model, model_name, optimizer_state_dict = load_vae_architecture(load_path, device, load_best=False,
                                                                        load_optimizer=True)
    else:
        model_type = select_vae_model(vae_name)
        model = model_type(config["model_parameters"], use_kld_warmup, kld_weight).to(device)
        optimizer_state_dict = None

    optimizer = optim.Adam(model.parameters(), lr=config["experiment_parameters"]["learning_rate"])

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # earlystopping = EarlyStopping('min', patience=30)

    # Check if VAE dir exists, if not, create it

    # Use a subfolder in the log for every dataset
    save_dir = os.path.join(config["logging_parameters"]["save_dir"], config["experiment_parameters"]["dataset"])

    experiment = Experiment(
        save_dir=save_dir,
        name=config["model_parameters"]["name"],
        debug=config["logging_parameters"]["debug"],  # Turns off logging if True
        create_git_tag=False,
        autosave=True
    )

    # Log hyperparameters to the tensorboard
    experiment.tag(config)

    # vae_dir = join(args.logdir, 'vae')
    # if not exists(vae_dir):
    #     mkdir(vae_dir)
    #     mkdir(join(vae_dir, 'samples'))

    # reload_file = join(vae_dir, 'best.tar')
    # if not args.noreload and exists(reload_file):
    #     state = torch.load(reload_file)
    #     print("Reloading model at epoch {}"
    #           ", with test error {}".format(
    #               state['epoch'],
    #               state['precision']))
    #     model.load_state_dict(state['state_dict'])
    #     optimizer.load_state_dict(state['optimizer'])
    #     scheduler.load_state_dict(state['scheduler'])
    #     earlystopping.load_state_dict(state['earlystopping'])

    current_best = None
    max_epochs = config["experiment_parameters"]["max_epochs"]

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
        logging.info(f"Started VAE training version_{training_version} for {max_epochs} epochs")

    for current_epoch in range(0, max_epochs):
        train(model, experiment, train_loader, optimizer, device, current_epoch, max_epochs)
        validation_loss = validate(model, experiment, val_loader, device, current_epoch, max_epochs)
        # scheduler.step(test_loss)
        # earlystopping.step(test_loss)

        # checkpointing
        if not experiment.debug:
            is_best = not current_best or validation_loss < current_best
            if is_best:
                current_best = validation_loss

            save_checkpoint({
                "epoch": current_epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
                # 'scheduler': scheduler.state_dict(),
                # 'earlystopping': earlystopping.state_dict()
            }, is_best, checkpoint_filename=checkpoint_filename, best_filename=best_model_filename)

            with torch.no_grad():
                sample_reconstructions = model.sample(batch_size, device).cpu()
                experiment.add_images("samples", sample_reconstructions, global_step=current_epoch)

        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(epoch))
        #     break

    if not experiment.debug:
        # Ensure everything is logged to the tensorboard
        experiment.flush()


if __name__ == "__main__":
    main()
