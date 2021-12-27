import logging
import os
from functools import partial
from os.path import join

import click
import torch
import torch.nn.functional as f
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from data.gui_dataset import GUISequenceDataset
from utils.misc import save_checkpoint, initialize_logger
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
# from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
# from utils.learning import ReduceLROnPlateau

# from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss


def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        obs, next_obs = [
            f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                       mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs


def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(data_loader, mdn_rnn, current_epoch, train, device):
    """ One pass through the data """
    # loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(data_loader.dataset), desc="Epoch {}".format(current_epoch))
    for i, data in enumerate(data_loader):
        observations, next_observations, rewards, actions = [d.to(device) for d in data]

        latent_obs, latent_next_obs = to_latent(observations, next_observations)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward)

            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
            loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
            gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


@click.command()
@click.option("-c", "--config", "config_path", type=str, required=True,
              help="Path to a YAML configuration containing training options")
def main(config_path: str):
    # parser = argparse.ArgumentParser("MDRNN training")
    # parser.add_argument('--logdir', type=str,
    #                     help="Where things are logged and models are loaded from.")
    # parser.add_argument('--noreload', action='store_true',
    #                     help="Do not reload if specified.")
    # parser.add_argument('--include_reward', action='store_true',
    #                     help="Add a reward modelisation term to the loss.")
    # args = parser.parse_args()
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    batch_size = config["experiment_parameters"]["batch_size"]
    sequence_length = config["experiment_parameters"]["sequence_length"]
    learning_rate = config["experiment_parameters"]["learning_rate"]
    max_epochs = config["experiment_parameters"]["max_epochs"]

    hidden_size = config["model_parameters"]["hidden_size"]

    dataset_name = config["experiment_parameters"]["dataset"]
    dataset_path = config["experiment_parameters"]["data_path"]

    num_workers = config["trainer_parameters"]["num_workers"]

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

    # constants
    # BSIZE = 16
    # SEQ_LEN = 32
    # epochs = 30

    vae_directory = config["vae_parameters"]["directory"]

    with open(os.path.join(vae_directory, "config.yaml")) as vae_config_file:
        vae_config = yaml.safe_load(vae_config_file)

    latent_size = vae_config["model_parameters"]["latent_size"]

    vae_name = vae_config["model_parameters"]["name"]

    vae = VAE(3, vae_config["model_parameters"]["latent_size"]).to(device)
    checkpoint = torch.load(os.path.join(vae_directory, "best.pt"), map_location=device)
    vae.load_state_dict(checkpoint["state_dict"])

    # Loading VAE
    # vae_file = join(args.logdir, 'vae', 'best.tar')
    # assert exists(vae_file), "No trained VAE in the logdir..."
    # state = torch.load(vae_file)
    # print("Loading VAE at epoch {} "
    #       "with test error {}".format(
    #     state['epoch'], state['precision']))

    # vae = VAE(3, LSIZE).to(device)
    # vae.load_state_dict(state['state_dict'])

    # Loading model
    # rnn_dir = join(args.logdir, 'mdrnn')
    # rnn_file = join(rnn_dir, 'best.tar')

    # if not exists(rnn_dir):
    #     mkdir(rnn_dir)

    action_size = 2

    mdn_rnn = MDRNN(latent_size, action_size, hidden_size, gaussians=5).to(device)
    optimizer = torch.optim.RMSprop(mdn_rnn.parameters(), lr=learning_rate, alpha=.9)

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

    set_range = transforms.Lambda(lambda x: 2 * x - 1.0)

    transformation_functions = transforms.Compose([
        transforms.Resize((vae_config["experiment_parameters"]["img_size"],
                           vae_config["experiment_parameters"]["img_size"])),
        transforms.ToTensor(),
        set_range
    ])

    train_dataset = GUISequenceDataset(dataset_path, train=True, sequence_length=sequence_length,
                                       transform=transformation_functions)
    test_dataset = GUISequenceDataset(dataset_path, train=False, sequence_length=sequence_length,
                                      transform=transformation_functions)

    additional_dataloader_args = {"num_workers": num_workers, "pin_memory": True}

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        **additional_dataloader_args
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **additional_dataloader_args
    )

    # Data Loading
    # transform = transforms.Lambda(
    #     lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
    # train_loader = DataLoader(
    #     RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=30),
    #     batch_size=BSIZE, num_workers=8, shuffle=True)
    # test_loader = DataLoader(
    #     RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=10),
    #     batch_size=BSIZE, num_workers=8)

    test = partial(data_pass, train=False, include_reward=args.include_reward)

    cur_best = None
    for e in range(max_epochs):
        mdn_rnn.train()
        data_pass(train_dataloader, mdn_rnn, e, train=True, device=device)

        mdn_rnn.eval()
        test_loss = data_pass(test_dataloader, mdn_rnn, e, train=False, device=device)
        # scheduler.step(test_loss)
        # earlystopping.step(test_loss)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
        checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
        save_checkpoint({
            "state_dict": mdrnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            "precision": test_loss,
            "epoch": e}, is_best, checkpoint_fname,
            rnn_file)

        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(e))
        #     break


if __name__ == "__main__":
    main()
