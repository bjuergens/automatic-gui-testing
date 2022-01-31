import json
import logging
import os

import click
import torch
import torch.utils.data
from torch import optim
from tqdm import tqdm

from data.dataset_implementations import get_vae_dataloader
from models import select_vae_model
from utils.logging.improved_summary_writer import ImprovedSummaryWriter
from utils.setup_utils import initialize_logger, load_yaml_config, set_seeds, get_device, save_yaml_config, pretty_json
from utils.training_utils import save_checkpoint, vae_transformation_functions, load_vae_architecture
from utils.training_utils.average_meter import AverageMeter


# from utils.misc import LSIZE, RED_SIZE
# WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
# from utils.learning import EarlyStopping
# from utils.learning import ReduceLROnPlateau
# from data.loaders import RolloutObservationDataset


NUMBER_OF_IMAGES_TO_LOG = 16


def train(model, summary_writer: ImprovedSummaryWriter, train_loader, optimizer, device, current_epoch, max_epochs,
          global_train_log_steps, debug: bool, scalar_log_frequency):
    model.train()

    total_loss_meter = AverageMeter("Loss", ":.4f")
    mse_loss_meter = AverageMeter("MSELoss", ":.4f")
    kld_loss_meter = AverageMeter("KLDLoss", ":.4f")

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), unit="batch",
                        desc=f"Epoch {current_epoch} - Train")

    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss, mse_loss, kld_loss = model.loss_function(data, recon_batch, mu, log_var)
        loss.backward()

        batch_size = data.size(0)
        total_loss_meter.update(loss.item(), batch_size)
        mse_loss_meter.update(mse_loss, batch_size)
        kld_loss_meter.update(kld_loss, batch_size)

        optimizer.step()

        if batch_idx % scalar_log_frequency == 0 or batch_idx == (len(train_loader) - 1):
            progress_bar.set_postfix_str(
                f"loss={total_loss_meter.avg:.4f} mse={mse_loss_meter.avg:.4f} kld={kld_loss_meter.avg:.4e}"
            )

            if not debug:
                summary_writer.add_scalar("loss", total_loss_meter.avg, global_step=global_train_log_steps)
                summary_writer.add_scalar("reconstruction_loss", mse_loss_meter.avg, global_step=global_train_log_steps)
                summary_writer.add_scalar("kld", kld_loss_meter.avg, global_step=global_train_log_steps)

                global_train_log_steps += 1

    progress_bar.close()

    if not debug:
        summary_writer.add_scalar("epoch_train_loss", total_loss_meter.avg, global_step=current_epoch)

    return global_train_log_steps


def validate(model, summary_writer: ImprovedSummaryWriter, val_loader, device, current_epoch, max_epochs,
             global_val_log_steps, debug: bool, scalar_log_frequency, image_epoch_log_frequency):
    model.eval()

    val_total_loss_meter = AverageMeter("Val_Loss", ":.4f")
    val_mse_loss_meter = AverageMeter("Val_MSELoss", ":.4f")
    val_kld_loss_meter = AverageMeter("Val_KLDLoss", ":.4e")

    logged_one_batch = False

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), unit="batch",
                        desc=f"Epoch {current_epoch} - Validation")

    for batch_idx, data in progress_bar:
        data = data.to(device)

        with torch.no_grad():
            recon_batch, mu, log_var = model(data)
            val_loss, val_mse_loss, val_kld_loss = model.loss_function(
                data, recon_batch, mu, log_var, train=False
            )

        batch_size = data.size(0)
        val_total_loss_meter.update(val_loss.item(), batch_size)
        val_mse_loss_meter.update(val_mse_loss, batch_size)
        val_kld_loss_meter.update(val_kld_loss, batch_size)

        if (not logged_one_batch
                and not debug
                and (current_epoch % image_epoch_log_frequency == 0 or current_epoch == (max_epochs - 1))):

            number_of_images = batch_size if batch_size < NUMBER_OF_IMAGES_TO_LOG else NUMBER_OF_IMAGES_TO_LOG

            summary_writer.add_images("originals", data[:number_of_images], global_step=current_epoch)
            summary_writer.add_images("reconstructions", recon_batch[:number_of_images], global_step=current_epoch)
            logged_one_batch = True

        if batch_idx % scalar_log_frequency == 0 or batch_idx == (len(val_loader) - 1):
            progress_bar.set_postfix_str(
                f"val_loss={val_total_loss_meter.avg:.4f} val_mse={val_mse_loss_meter.avg:.4f} "
                f"val_kld={val_kld_loss_meter.avg:.4e}"
            )

            if not debug:
                summary_writer.add_scalar("val_loss", val_total_loss_meter.avg, global_step=global_val_log_steps)
                summary_writer.add_scalar("val_reconstruction_loss", val_mse_loss_meter.avg,
                                          global_step=global_val_log_steps)
                summary_writer.add_scalar("val_kld", val_kld_loss_meter.avg, global_step=global_val_log_steps)

                global_val_log_steps += 1

    progress_bar.close()

    if not debug:
        summary_writer.add_scalar("epoch_val_loss", val_total_loss_meter.avg, global_step=current_epoch)

    return val_total_loss_meter.avg, global_val_log_steps


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

    batch_size = config["experiment_parameters"]["batch_size"]
    manual_seed = config["experiment_parameters"]["manual_seed"]

    dataset_name = config["experiment_parameters"]["dataset"]
    dataset_path = config["experiment_parameters"]["dataset_path"]

    img_size = config["experiment_parameters"]["img_size"]
    learning_rate = config["experiment_parameters"]["learning_rate"]

    number_of_workers = config["trainer_parameters"]["num_workers"]
    gpu_id = config["trainer_parameters"]["gpu"]

    # VAE configuration
    vae_name = config["model_parameters"]["name"]

    max_epochs = config["experiment_parameters"]["max_epochs"]
    debug = config["logging_parameters"]["debug"]

    scalar_log_frequency = config["logging_parameters"]["scalar_log_frequency"]
    image_epoch_log_frequency = config["logging_parameters"]["image_epoch_log_frequency"]

    set_seeds(manual_seed)
    device = get_device(gpu_id)

    transformation_functions = vae_transformation_functions(img_size)

    additional_dataloader_kwargs = {"num_workers": number_of_workers, "pin_memory": True, "drop_last": True}

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
        model = model_type(config["model_parameters"]).to(device)
        optimizer_state_dict = None

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # earlystopping = EarlyStopping('min', patience=30)

    # Check if VAE dir exists, if not, create it

    # Use a subfolder in the log for every dataset
    save_dir = os.path.join(config["logging_parameters"]["save_dir"], config["experiment_parameters"]["dataset"])
    global_train_log_steps = 0
    global_val_log_steps = 0

    if not debug:
        summary_writer = ImprovedSummaryWriter(
            log_dir=save_dir
        )

        # Log hyperparameters to the tensorboard
        summary_writer.add_text("Hyperparameters", pretty_json(config), global_step=0)

        log_dir = summary_writer.get_logdir()
        best_model_filename = os.path.join(log_dir, "best.pt")
        checkpoint_filename = os.path.join(log_dir, "checkpoint.pt")

        save_yaml_config(os.path.join(log_dir, "config.yaml"), config)

        logging.info(f"Started VAE training version_{summary_writer.version_number} for {max_epochs} epochs")
    else:
        summary_writer = None
        # Enables debugging of the gradient calculation, shows where errors/NaN etc. occur
        torch.autograd.set_detect_anomaly(True)

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
    validation_loss = None

    for current_epoch in range(0, max_epochs):
        global_train_log_steps = train(model, summary_writer, train_loader, optimizer, device, current_epoch,
                                       max_epochs, global_train_log_steps, debug, scalar_log_frequency)
        validation_loss, global_val_log_steps = validate(model, summary_writer, val_loader, device, current_epoch,
                                                         max_epochs, global_val_log_steps, debug, scalar_log_frequency,
                                                         image_epoch_log_frequency)
        # scheduler.step(test_loss)
        # earlystopping.step(test_loss)

        # checkpointing
        if not debug:
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

            if current_epoch % image_epoch_log_frequency == 0 or current_epoch == (max_epochs - 1):
                number_of_images = batch_size if batch_size < NUMBER_OF_IMAGES_TO_LOG else NUMBER_OF_IMAGES_TO_LOG
                model.eval()
                with torch.no_grad():
                    sample_reconstructions = model.sample(number_of_images, device).cpu()
                    summary_writer.add_images("samples", sample_reconstructions, global_step=current_epoch)

        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(epoch))
        #     break

    if not debug:
        # Use prefix m for model_parameters to avoid possible reassignment of a hparam when combining with
        # experiment_parameters
        model_params = {f"m_{k}": v for k, v in config["model_parameters"].items()}

        for k, v in model_params.items():
            if isinstance(v, list):
                model_params[k] = ", ".join(str(x) for x in v)

        exp_params = {f"e_{k}": v for k, v in config["experiment_parameters"].items()}

        hparams = {**model_params, **exp_params}

        summary_writer.add_hparams(
            hparams,
            {"hparams/val_loss": validation_loss, "hparams/best_val_loss": current_best},
            run_name=f"hparams"  # Since we use one folder per vae training run we can use a fix name here
        )
        # Ensure everything is logged to the tensorboard
        summary_writer.flush()


if __name__ == "__main__":
    main()
