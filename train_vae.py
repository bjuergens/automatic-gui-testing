import logging
import os
import sys

import click
# noinspection PyUnresolvedReferences
import comet_ml  # Needs to be imported __before__ torch
import torch
import torch.utils.data
from torch import optim
from tqdm import tqdm

from data.dataset_implementations import get_vae_dataloader
from models import select_vae_model
from utils.logging.improved_summary_writer import ImprovedSummaryWriter, ExistingImprovedSummaryWriter
from utils.setup_utils import initialize_logger, load_yaml_config, set_seeds, get_device, save_yaml_config, pretty_json
from utils.training_utils import save_checkpoint, vae_transformation_functions
from utils.training_utils.average_meter import AverageMeter
from utils.training_utils.training_utils import get_dataset_mean_std, load_vae_architecture

NUMBER_OF_IMAGES_TO_LOG = 16


def train(model, summary_writer: ImprovedSummaryWriter, train_loader, optimizer, device, current_epoch,
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

            # Occasionally check if NaN are produced in training, if so stop it
            if torch.isnan(loss).any():
                summary_writer.flush()
                sys.exit("During VAE training a NaN was produced, stopping training now")

        global_train_log_steps += 1

    progress_bar.close()

    if not debug:
        summary_writer.add_scalar("epoch_train_loss", total_loss_meter.avg, global_step=current_epoch)

    return global_train_log_steps


def compute_test_performance(model, existing_summary_writer, test_loader, device, scalar_log_frequency):
    model.eval()

    test_total_loss_meter = AverageMeter("Test_Loss", ":.4f")
    test_mse_loss_meter = AverageMeter("Test_MSELoss", ":.4f")
    test_kld_loss_meter = AverageMeter("Test_KLDLoss", ":.4e")

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), unit="batch",
                        desc=f"Test Data")

    for batch_idx, data in progress_bar:
        data = data.to(device)

        with torch.no_grad():
            recon_batch, mu, log_var = model(data)
            test_loss, test_mse_loss, test_kld_loss = model.loss_function(
                data, recon_batch, mu, log_var, train=False
            )

            batch_size = data.size(0)
            test_total_loss_meter.update(test_loss.item(), batch_size)
            test_mse_loss_meter.update(test_mse_loss, batch_size)
            test_kld_loss_meter.update(test_kld_loss, batch_size)

        if batch_idx % scalar_log_frequency == 0 or batch_idx == (len(test_loader) - 1):
            progress_bar.set_postfix_str(
                f"test_loss={test_total_loss_meter.avg:.4f} test_mse={test_mse_loss_meter.avg:.4f} "
                f"test_kld={test_kld_loss_meter.avg:.4e}"
            )

            existing_summary_writer.add_scalar("test_loss", test_total_loss_meter.avg, global_step=batch_idx)
            existing_summary_writer.add_scalar("test_reconstruction_loss", test_mse_loss_meter.avg,
                                               global_step=batch_idx)
            existing_summary_writer.add_scalar("test_kld", test_kld_loss_meter.avg, global_step=batch_idx)

    progress_bar.close()

    existing_summary_writer.add_scalar("epoch_test_loss", test_total_loss_meter.avg, global_step=0)


def validate(model, summary_writer: ImprovedSummaryWriter, val_loader, device, current_epoch, max_epochs,
             global_val_log_steps, debug: bool, scalar_log_frequency, image_epoch_log_frequency,
             denormalize_with_mean_and_std_necessary, dataset_mean, dataset_std):
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

            if denormalize_with_mean_and_std_necessary:
                summary_writer.add_images("originals", (data[:number_of_images] * dataset_std) + dataset_mean,
                                          global_step=current_epoch)
                summary_writer.add_images("reconstructions",
                                          (recon_batch[:number_of_images] * dataset_std) + dataset_mean,
                                          global_step=current_epoch)
            else:
                summary_writer.add_images("originals", model.denormalize(data[:number_of_images]),
                                          global_step=current_epoch)
                summary_writer.add_images("reconstructions", model.denormalize(recon_batch[:number_of_images]),
                                          global_step=current_epoch)
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
@click.option("-c", "--config", "config_path", type=str,
              help="Path to a YAML configuration containing training options")
@click.option("-l", "--load", "load_path", type=str,
              help=("Path to a previous training, from which training shall continue (will create a new experiment "
                    "directory)"))
@click.option("--disable-comet/--no-disable-comet", type=bool, default=False,
              help="Disable logging to Comet (automatically disabled when API key is not provided in home folder)")
@click.option("--test-data/--no-test-data", type=bool, default=False,
              help="Loads a VAE, computes the performance on test set and logs it to an existing Comet experiment")
@click.option("--test-vae-dir", type=str, default=None, help="Path to a trained VAE directory, which shall be loaded to"
                                                             "log the test performance")
@click.option("--comet-exp-id", type=str, default=None, help="Existing Comet experiment ID to which the test"
                                                             "performance shall be logged")
@click.option("--test-num-workers", type=int, default=8, help="Number of worker processes during test data computation")
@click.option("--test-gpu", type=int, default=-1, help="Number of GPU to be used for test data computation, -1 is CPU")
def main(config_path: str, load_path: str, disable_comet: bool, test_data: bool, test_vae_dir: str, comet_exp_id: str,
         test_num_workers: int, test_gpu: int):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    if not test_data:
        assert config_path is not None, "Path to a config required when training a VAE"

        config = load_yaml_config(config_path)

        batch_size = config["experiment_parameters"]["batch_size"]
        manual_seed = config["experiment_parameters"]["manual_seed"]

        dataset_name = config["experiment_parameters"]["dataset"]
        dataset_path = config["experiment_parameters"]["dataset_path"]

        img_size = config["experiment_parameters"]["img_size"]
        learning_rate = config["experiment_parameters"]["learning_rate"]

        try:
            lr_scheduler_dict = config["lr_scheduler"]
            use_lr_scheduler = lr_scheduler_dict["use_lr_scheduler"]
        except KeyError:
            lr_scheduler_dict = None
            use_lr_scheduler = False

        number_of_workers = config["trainer_parameters"]["num_workers"]
        gpu_id = config["trainer_parameters"]["gpu"]

        # VAE configuration
        vae_name = config["model_parameters"]["name"]

        max_epochs = config["experiment_parameters"]["max_epochs"]
        debug = config["logging_parameters"]["debug"]

        scalar_log_frequency = config["logging_parameters"]["scalar_log_frequency"]
        image_epoch_log_frequency = config["logging_parameters"]["image_epoch_log_frequency"]
        save_model_checkpoints = config["logging_parameters"]["save_model_checkpoints"]

        set_seeds(manual_seed)
        device = get_device(gpu_id)

        transformation_functions = vae_transformation_functions(img_size, dataset_name,
                                                                config["model_parameters"]["output_activation_function"])
        dataset_mean, dataset_std = get_dataset_mean_std(dataset_name)

        if dataset_mean is not None and dataset_std is not None:
            denormalize_with_mean_and_std_necessary = True
            dataset_mean = torch.tensor(dataset_mean).view(1, len(dataset_mean), 1, 1).to(device)
            dataset_std = torch.tensor(dataset_std).view(1, len(dataset_std), 1, 1).to(device)
        else:
            denormalize_with_mean_and_std_necessary = False

        additional_dataloader_kwargs = {"num_workers": number_of_workers, "pin_memory": True, "drop_last": False}

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
            raise RuntimeError("Do not use the --load option it is not working properly (config mismatch etc.)")
            # model, model_name, optimizer_state_dict = load_vae_architecture(load_path, device, load_best=False,
            #                                                                 load_optimizer=True)
        else:
            model_type = select_vae_model(vae_name)
            model = model_type(config["model_parameters"]).to(device)
            optimizer_state_dict = None

        try:
            optimizer_name = config["experiment_parameters"]["optimizer"]
        except KeyError:
            # Default Optimizer is adam
            optimizer_name = "adam"

        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "adamax":
            optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
        else:
            raise RuntimeError(f"Optimizer '{optimizer_name}' unknown")

        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        if use_lr_scheduler:
            scheduler_params = {
                "mode": lr_scheduler_dict["mode"],
                "patience": lr_scheduler_dict["patience"],
                "factor": lr_scheduler_dict["factor"],
                "threshold": lr_scheduler_dict["threshold"],
                "threshold_mode": lr_scheduler_dict["threshold_mode"],
            }

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                **scheduler_params
            )
        else:
            scheduler_params = None
            scheduler = None

        # Use a subfolder in the log for every dataset
        save_dir = os.path.join(config["logging_parameters"]["save_dir"], config["experiment_parameters"]["dataset"])
        global_train_log_steps = 0
        global_val_log_steps = 0

        if not debug:
            summary_writer = ImprovedSummaryWriter(
                log_dir=save_dir,
                comet_config={
                    "project_name": "world-models/vae",
                    "disabled": disable_comet
                }
            )

            # Log hyperparameters to the tensorboard
            summary_writer.add_text("Hyperparameters", pretty_json(config), global_step=0)

            # Unfortunately tensorboardX does not expose this functionality and name cannot be set in constructor
            if not disable_comet:
                # noinspection PyProtectedMember
                summary_writer._get_comet_logger()._experiment.set_name(f"version_{summary_writer.version_number}")

            log_dir = summary_writer.get_logdir()
            best_model_filename = os.path.join(log_dir, "best.pt")
            checkpoint_filename = os.path.join(log_dir, "checkpoint.pt")

            save_yaml_config(os.path.join(log_dir, "config.yaml"), config)

            logging.info(f"Started VAE training version_{summary_writer.version_number} for {max_epochs} epochs")
        else:
            summary_writer = None
            # Enables debugging of the gradient calculation, shows where errors/NaN etc. occur
            torch.autograd.set_detect_anomaly(True)

        current_best = None
        validation_loss = None

        for current_epoch in range(0, max_epochs):
            global_train_log_steps = train(model, summary_writer, train_loader, optimizer, device, current_epoch,
                                           global_train_log_steps, debug, scalar_log_frequency)
            validation_loss, global_val_log_steps = validate(model, summary_writer, val_loader, device, current_epoch,
                                                             max_epochs, global_val_log_steps, debug, scalar_log_frequency,
                                                             image_epoch_log_frequency,
                                                             denormalize_with_mean_and_std_necessary, dataset_mean,
                                                             dataset_std)

            if use_lr_scheduler:
                scheduler.step(validation_loss)

            # checkpointing
            if not debug:
                is_best = not current_best or validation_loss < current_best
                if is_best:
                    current_best = validation_loss

                if save_model_checkpoints:

                    state_dicts = {
                        "epoch": current_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }

                    if use_lr_scheduler:
                        state_dicts["scheduler"] = scheduler.state_dict()

                    # noinspection PyUnboundLocalVariable
                    save_checkpoint(
                        state_dicts,
                        is_best,
                        checkpoint_filename=checkpoint_filename,
                        best_filename=best_model_filename
                    )

                if current_epoch % image_epoch_log_frequency == 0 or current_epoch == (max_epochs - 1):
                    number_of_images = batch_size if batch_size < NUMBER_OF_IMAGES_TO_LOG else NUMBER_OF_IMAGES_TO_LOG
                    model.eval()
                    with torch.no_grad():
                        sample_reconstructions = model.sample(number_of_images, device).cpu()

                        if denormalize_with_mean_and_std_necessary:
                            summary_writer.add_images("samples", (sample_reconstructions * dataset_std) + dataset_mean,
                                                      global_step=current_epoch)
                        else:
                            summary_writer.add_images("samples", model.denormalize(sample_reconstructions),
                                                      global_step=current_epoch)

        if not debug:
            # Use prefix m for model_parameters to avoid possible reassignment of a hparam when combining with
            # experiment_parameters
            model_params = {f"m_{k}": v for k, v in config["model_parameters"].items()}

            for k, v in model_params.items():
                if isinstance(v, list):
                    model_params[k] = ", ".join(str(x) for x in v)

            exp_params = {f"e_{k}": v for k, v in config["experiment_parameters"].items()}

            if use_lr_scheduler:
                modified_scheduler_params = {f"lr_{k}": v for k, v in scheduler_params.items()}
                hparams = {**model_params, **exp_params, **modified_scheduler_params}
            else:
                hparams = {**model_params, **exp_params}

            summary_writer.add_hparams(
                hparams,
                {"hparams/val_loss": validation_loss, "hparams/best_val_loss": current_best},
                name="hparams"
            )
            # Ensure everything is logged to the tensorboard
            summary_writer.flush()
    else:
        existing_summary_writer = ExistingImprovedSummaryWriter(experiment_key=comet_exp_id)

        assert existing_summary_writer.exp.name == test_vae_dir.split("/")[-1], ("Name in Comet experiment and name "
                                                                                 "from log directory do not match")

        test_config_path = os.path.join(test_vae_dir, "config.yaml")
        config = load_yaml_config(test_config_path)

        dataset_name = config["experiment_parameters"]["dataset"]
        dataset_path = config["experiment_parameters"]["dataset_path"]
        batch_size = config["experiment_parameters"]["batch_size"]
        img_size = config["experiment_parameters"]["img_size"]
        scalar_log_frequency = config["logging_parameters"]["scalar_log_frequency"]

        transformation_functions = vae_transformation_functions(
            img_size,
            dataset_name,
            config["model_parameters"]["output_activation_function"]
        )

        additional_dataloader_kwargs = {"num_workers": test_num_workers, "pin_memory": True, "drop_last": False}

        test_loader = get_vae_dataloader(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split="test",
            transformation_functions=transformation_functions,
            batch_size=batch_size,
            shuffle=False,
            **additional_dataloader_kwargs
        )

        device = get_device(test_gpu)

        vae, vae_name = load_vae_architecture(test_vae_dir, device, load_best=True)
        vae.eval()

        logging.info(f"Starting computation of test performance for VAE {test_vae_dir} and logging to Comet"
                     f"experiment {comet_exp_id}")

        compute_test_performance(
            model=vae,
            existing_summary_writer=existing_summary_writer,
            test_loader=test_loader,
            device=device,
            scalar_log_frequency=scalar_log_frequency
        )

        existing_summary_writer.close()


if __name__ == "__main__":
    main()
