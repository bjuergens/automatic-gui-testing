# import argparse
import os

import click
import torch
import torch.utils.data
import yaml
from test_tube import Experiment
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from data.gui_dataset import GUIDataset
from models.vae import VAE

from utils.misc import save_checkpoint
# from utils.misc import LSIZE, RED_SIZE
# WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
# from utils.learning import EarlyStopping
# from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch, model, train_loader, device, optimizer):
    """ One training epoch """
    model.train()
    # dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def validate(model, val_loader, device):
    """ One test epoch """
    model.eval()
    # dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


@click.command()
@click.option("-c", "--config", "config_path", type=str, required=True,
              help="Path to a YAML configuration containing training options")
def main(config_path: str):
    # parser = argparse.ArgumentParser(description='Train the VAE (V model of the world model)')
    # parser.add_argument('--batch-size', type=int, default=32, metavar='N',
    #                     help='Batch size for training (default: 32)')
    # parser.add_argument('--epochs', type=int, default=1000, metavar='N',
    #                     help='Number of epochs to train (default: 1000)')
    # parser.add_argument('--logdir', type=str, help='Directory where results are logged')
    # parser.add_argument('--noreload', action='store_true',
    #                     help='Best model is not reloaded if specified')
    # parser.add_argument('--nosamples', action='store_true',
    #                     help='Does not save samples during training if specified')

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # SIZE_HEIGHT, SIZE_WIDTH = 224, 224
    # LATENT_SIZE = 64

    # args = parser.parse_args()
    torch.manual_seed(123)

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

    # transform_test = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((RED_SIZE, RED_SIZE)),
    #     transforms.ToTensor(),
    # ])


    #
    # dataset_train = RolloutObservationDataset('datasets/carracing',
    #                                           transform_train, train=True)
    # dataset_test = RolloutObservationDataset('datasets/carracing',
    #                                          transform_test, train=False)

    additional_dataloader_args = {'num_workers': config["trainer_parameters"]["num_workers"], 'pin_memory': True}
    batch_size = config["experiment_parameters"]["batch_size"]

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

    model = VAE(img_channels=3, latent_size=config["model_parameters"]["latent_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["experiment_parameters"]["learning_rate"])
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # earlystopping = EarlyStopping('min', patience=30)

    # Check if VAE dir exists, if not, create it

    # Use a subfolder in the log for every dataset
    save_dir = os.path.join(config["logging_parameters"]["save_dir"], config["experiment_parameters"]["dataset"])
    os.makedirs(save_dir, exist_ok=True)

    # TODO figure out if autosave is needed
    experiment = Experiment(
        save_dir=save_dir,
        name=config["model_parameters"]["name"],
        debug=config["logging_parameters"]["debug"],  # Turns off logging if True
        create_git_tag=False
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
    epochs = config["trainer_parameters"]["max_epochs"]

    log_dir = experiment.get_logdir().split("tf")[0]

    os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, device, optimizer)
        validation_loss = validate(model, val_loader, device)
        # scheduler.step(test_loss)
        # earlystopping.step(test_loss)

        # checkpointing
        best_filename = os.path.join(log_dir, 'best.tar')
        filename = os.path.join(log_dir, 'checkpoint.tar')
        is_best = not current_best or validation_loss < current_best
        if is_best:
            current_best = validation_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': validation_loss,
            'optimizer': optimizer.state_dict()
            # 'scheduler': scheduler.state_dict(),
            # 'earlystopping': earlystopping.state_dict()
        }, is_best, filename, best_filename)

        with torch.no_grad():
            sample = torch.randn(10, config["model_parameters"]["latent_size"]).to(device)
            reconstruction = model.decoder(sample).cpu()
            save_image(reconstruction, os.path.join(log_dir, 'samples/sample_' + str(epoch) + '.png'))

        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(epoch))
        #     break


if __name__ == '__main__':
    main()
