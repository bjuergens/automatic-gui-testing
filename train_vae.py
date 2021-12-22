import os

import click
import torch
import torch.utils.data
import yaml
from test_tube import Experiment
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from data.gui_dataset import GUIDataset
from models.vae import VAE
from utils.misc import save_checkpoint


# from utils.misc import LSIZE, RED_SIZE
# WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
# from utils.learning import EarlyStopping
# from utils.learning import ReduceLROnPlateau
# from data.loaders import RolloutObservationDataset


def loss_function(experiment: Experiment, x: torch.Tensor, reconstruction_x: torch.Tensor, mu: torch.Tensor,
                  log_var: torch.Tensor, kld_weight: float, current_epoch: int, max_epochs: int,
                  is_train: bool = True) -> torch.Tensor:
    # MSE
    batch_dim = x.size(0)
    reconstruction_loss = F.mse_loss(x, reconstruction_x, reduction="sum") / batch_dim

    # KLD
    kld_warmup_term = current_epoch / max_epochs

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Take also the mean over the batch_dim (outermost function call)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

    loss = reconstruction_loss + kld_weight * kld_warmup_term * kld_loss

    # .item() is important as it extracts a float, otherwise the tensors would be held in memory and never freed
    if is_train:
        experiment.log({
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld": kld_loss.item(),
            "mu": torch.mean(mu).item(),
            "log_var": torch.mean(log_var).item(),
            "var": torch.mean(log_var.exp()).item()
        })
    else:
        experiment.log({
            "val_loss": loss.item(),
            "val_reconstruction_loss": reconstruction_loss.item(),
            "val_kld": kld_loss.item(),
            "val_mu": torch.mean(mu).item(),
            "val_log_var": torch.mean(log_var).item(),
            "val_var": torch.mean(log_var.exp()).item()
        })

    return loss


def train(model, experiment, train_loader, optimizer, device, current_epoch, max_epochs, kld_weight):
    """ One training epoch """
    model.train()
    # dataset_train.load_next_buffer()
    train_loss = 0
    train_mu = None
    train_log_var = None

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), unit="batch",
                        desc=f"Epoch {current_epoch} - Train")

    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(experiment, data, recon_batch, mu, logvar, kld_weight, current_epoch, max_epochs)
        loss.backward()
        train_loss += loss.item() * data.size(0)
        optimizer.step()

        if train_mu is None and train_log_var is None:
            train_mu = mu
            train_log_var = logvar
        else:
            train_mu = torch.cat([train_mu, mu], dim=0)
            train_log_var = torch.cat([train_log_var, logvar], dim=0)

        # if batch_idx % 20 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         current_epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

        # pbar.set_description("Loss {:.4f}".format(loss.item()))
        progress_bar.set_postfix({"loss": loss.item()})
    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #     current_epoch, train_loss / len(train_loader.dataset)))

    experiment.log({
        "epoch_train_loss": train_loss / len(train_loader.dataset),
        "epoch_mu": train_mu.mean().item(),
        "epoch_var": train_log_var.exp().mean().item()
    })


def validate(model, experiment: Experiment, val_loader, device, current_epoch, max_epochs, kld_weight):
    """ One test epoch """
    model.eval()

    val_loss_sum = 0

    logged_one_batch = False

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), unit="batch",
                        desc=f"Epoch {current_epoch} - Validation")

    for _, data in progress_bar:
        with torch.no_grad():
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            val_loss = loss_function(experiment, data, recon_batch, mu, log_var, kld_weight, current_epoch, max_epochs,
                                     is_train=False).item()

            progress_bar.set_postfix({"val_loss": val_loss})

            val_loss_sum += val_loss

        if not logged_one_batch and not experiment.debug:
            experiment.add_images("originals", data, global_step=current_epoch)
            experiment.add_images("reconstructions", recon_batch, global_step=current_epoch)
            logged_one_batch = True

    val_loss_sum /= len(val_loader.dataset)
    # print('====> Test set loss: {:.4f}'.format(test_loss))

    model.train()

    return val_loss_sum


@click.command()
@click.option("-c", "--config", "config_path", type=str, required=True,
              help="Path to a YAML configuration containing training options")
def main(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    latent_size = config["model_parameters"]["latent_size"]

    kld_weight = config["experiment_parameters"]["kld_weight"]
    batch_size = config["experiment_parameters"]["batch_size"]
    manual_seed = config["experiment_parameters"]["manual_seed"]

    torch.manual_seed(manual_seed)

    # Fix numeric divergence due to bug in Cudnn
    # Also: Seems to search for the best algorithm to use; don't use if the input size changes a lot then it hurts
    # performance
    torch.backends.cudnn.benchmark = True

    if config["trainer_parameters"]["gpu"] >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['trainer_parameters']['gpu']}")
    else:
        device = torch.device("cpu")

    set_range = transforms.Lambda(lambda x: 2 * x - 1.0)

    transformation_functions = transforms.Compose([
        transforms.Resize((config["experiment_parameters"]["img_size"], config["experiment_parameters"]["img_size"])),
        transforms.ToTensor(),
        set_range
    ])

    if config["experiment_parameters"]["dataset"] == "gui-dataset":
        data_path = config["experiment_parameters"]["data_path"]
        train_dataset = GUIDataset(
            data_path,
            split="train",
            transform=transformation_functions
        )
        val_dataset = GUIDataset(
            data_path,
            split="val",
            transform=transformation_functions
        )
    else:
        raise RuntimeError("Currently only 'gui-dataset' supported as the dataset")

    additional_dataloader_args = {'num_workers': config["trainer_parameters"]["num_workers"], 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **additional_dataloader_args
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **additional_dataloader_args
    )

    model = VAE(img_channels=3, latent_size=latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["experiment_parameters"]["learning_rate"])
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # earlystopping = EarlyStopping('min', patience=30)

    # Check if VAE dir exists, if not, create it

    # Use a subfolder in the log for every dataset
    save_dir = os.path.join(config["logging_parameters"]["save_dir"], config["experiment_parameters"]["dataset"])
    os.makedirs(save_dir, exist_ok=True)

    experiment = Experiment(
        save_dir=save_dir,
        name=config["model_parameters"]["name"],
        debug=config["logging_parameters"]["debug"],  # Turns off logging if True
        create_git_tag=False,
        autosave=True
    )

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
    max_epochs = config["trainer_parameters"]["max_epochs"]

    if not experiment.debug:
        log_dir = experiment.get_logdir().split("tf")[0]
        best_model_filename = os.path.join(log_dir, "best.pt")
        checkpoint_filename = os.path.join(log_dir, "checkpoint.pt")

    for current_epoch in range(0, max_epochs):
        train(model, experiment, train_loader, optimizer, device, current_epoch, max_epochs, kld_weight)
        validation_loss = validate(model, experiment, val_loader, device, current_epoch, max_epochs, kld_weight)
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
            }, is_best, checkpoint_filename, best_model_filename)

            with torch.no_grad():
                sample = torch.randn(batch_size, latent_size).to(device)
                sample_reconstructions = model.decoder(sample).cpu()
                experiment.add_images("samples", sample_reconstructions, global_step=current_epoch)

        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(epoch))
        #     break


if __name__ == '__main__':
    main()
