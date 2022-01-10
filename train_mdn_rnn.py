import logging
import os

import click
import torch
import torch.nn.functional as f
import yaml
from test_tube import Experiment
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.gui_dataset import GUISequenceDataset, GUIMultipleSequencesDataset, GUISequenceBatchSampler
from models.mdrnn import MDRNN, gmm_loss
# from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from utils.misc import save_checkpoint, initialize_logger


# from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
# from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
# from utils.learning import ReduceLROnPlateau


def to_latent(obs, next_obs, vae, latent_size):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        # obs, next_obs = [
        #     f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
        #                mode='bilinear', align_corners=True)
        #     for x in (obs, next_obs)]
        batch_size = obs.size(0)
        sequence_length = obs.size(1)

        obs = obs.view(-1, 3, obs.size(3), obs.size(4))
        next_obs = next_obs.view(-1, 3, next_obs.size(3), next_obs.size(4))

        obs_mu, obs_log_var = vae(obs)[1:]
        next_obs_mu, next_obs_log_var = vae(next_obs)[1:]

        latent_obs = torch.randn_like(obs_mu).mul(torch.exp(0.5 * obs_log_var)).add_(obs_mu)
        latent_next_obs = torch.randn_like(next_obs_mu).mul(torch.exp(0.5 * next_obs_log_var)).add_(next_obs_mu)

        latent_obs = latent_obs.view(batch_size, sequence_length, latent_size)
        latent_next_obs = latent_next_obs.view(batch_size, sequence_length, latent_size)

        # latent_obs, latent_next_obs = [
        #     (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(batch_size, sequence_length, latent_size)
        #     for x_mu, x_logsigma in
        #     [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs


def get_loss(mdn_rnn, latent_obs, action, reward, latent_next_obs):
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
    latent_obs, latent_next_obs, reward, action = [d.transpose(1, 0) for d in [latent_obs, latent_next_obs, reward, action]]

    mus, sigmas, log_pi, rs = mdn_rnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, log_pi)
    # bce = f.binary_cross_entropy_with_logits(ds, terminal)

    mse = f.mse_loss(rs, reward)
    # scale = LSIZE + 2

    # loss = (gmm + bce + mse) / scale
    loss = (gmm + mse)

    return loss, gmm, mse


def data_pass(mdn_rnn, vae, experiment, optimizer, data_loader: DataLoader, latent_size, device: torch.device,
              current_epoch: int, train: bool):
    """ One pass through the data """
    # loader.dataset.load_next_buffer()

    if train:
        mdn_rnn.train()
        loss_key = "loss"
        gmm_key = "gmm"
        mse_key = "mse"
    else:
        mdn_rnn.eval()
        loss_key = "val_loss"
        gmm_key = "val_gmm"
        mse_key = "val_mse"

    cum_loss = 0
    cum_gmm = 0
    # cum_bce = 0
    cum_mse = 0

    old_dataset_index = None

    log_interval = 20

    pbar = tqdm(total=len(data_loader.dataset), desc="Epoch {}".format(current_epoch))
    for i, data in enumerate(data_loader):
        observations, next_observations, rewards, actions = [d.to(device) for d in data[0]]
        dataset_indices: torch.Tensor = data[1]
        current_dataset_index = dataset_indices[0]
        assert all(dataset_indices == current_dataset_index)

        if old_dataset_index is None or old_dataset_index != current_dataset_index:
            old_dataset_index = current_dataset_index
            mdn_rnn.initialize_hidden()

        batch_size = observations.size(0)

        latent_obs, latent_next_obs = to_latent(observations, next_observations, vae, latent_size)

        if train:
            optimizer.zero_grad()
            loss, gmm, mse = get_loss(mdn_rnn, latent_obs, actions, rewards, latent_next_obs)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, gmm, mse = get_loss(mdn_rnn, latent_obs, actions, rewards, latent_next_obs)

        cum_loss += loss.item() * batch_size
        cum_gmm += gmm.item() * batch_size
        # cum_bce += bce * batch_size
        cum_mse += mse.item() * batch_size

        # TODO gmm divide by latent_size correct? Was done by previous authors
        pbar.set_postfix_str("loss={loss:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
            loss=cum_loss / ((i + 1) * batch_size),
            gmm=cum_gmm / latent_size / ((i + 1) * batch_size), mse=cum_mse / ((i + 1) * batch_size)))
        pbar.update(batch_size)

        if i % log_interval == 0:
            experiment.log({
                loss_key: loss.item(),
                gmm_key: gmm.item(),
                mse_key: mse.item()
            })

    pbar.close()

    experiment.log({
        f"epoch_{loss_key}": cum_loss / len(data_loader.dataset),
        f"epoch_{gmm_key}": cum_gmm / len(data_loader.dataset),
        f"epoch_{mse_key}": cum_mse / len(data_loader.dataset),
    })

    return cum_loss * batch_size / len(data_loader.dataset)


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
    # torch.autograd.set_detect_anomaly(True)
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

    base_save_dir = config["logging_parameters"]["base_save_dir"]
    mdn_rnn_name = config["model_parameters"]["name"]

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

    mdn_rnn = MDRNN(latent_size, action_size, hidden_size, gaussians=5, batch_size=batch_size, device=device).to(device)
    # optimizer = torch.optim.RMSprop(mdn_rnn.parameters(), lr=learning_rate, alpha=.9)
    optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=learning_rate)

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

    if dataset_name == "gui_multiple_sequences":
        train_dataset = GUIMultipleSequencesDataset(dataset_path, train=True, sequence_length=sequence_length,
                                                    transform=transformation_functions)
        test_dataset = GUIMultipleSequencesDataset(dataset_path, train=False, sequence_length=sequence_length,
                                                   transform=transformation_functions)
    else:
        train_dataset = GUISequenceDataset(dataset_path, train=True, sequence_length=sequence_length,
                                           transform=transformation_functions)
        test_dataset = GUISequenceDataset(dataset_path, train=False, sequence_length=sequence_length,
                                          transform=transformation_functions)

    additional_dataloader_args = {"num_workers": num_workers, "pin_memory": True}

    custom_sampler_train = GUISequenceBatchSampler(train_dataset, batch_size=batch_size, drop_last=True)
    custom_sampler_test = GUISequenceBatchSampler(test_dataset, batch_size=batch_size, drop_last=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=custom_sampler_train,
        **additional_dataloader_args
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=custom_sampler_test,
        **additional_dataloader_args
    )

    save_dir = os.path.join(base_save_dir, dataset_name)

    experiment = Experiment(
        save_dir=save_dir,
        name=mdn_rnn_name,
        debug=config["logging_parameters"]["debug"],  # Turns off logging if True
        create_git_tag=False,
        autosave=True
    )

    if not experiment.debug:
        log_dir = experiment.get_logdir().split("tf")[0]
        best_model_filename = os.path.join(log_dir, "best.pt")
        checkpoint_filename = os.path.join(log_dir, "checkpoint.pt")

        with open(os.path.join(log_dir, "config.yaml"), "w") as file:
            yaml.safe_dump(config, file, default_flow_style=False)

    training_version = experiment.version
    if training_version is not None:
        logging.info(f"Started MDN-RNN training version_{training_version} for {max_epochs}")

    # Data Loading
    # transform = transforms.Lambda(
    #     lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
    # train_loader = DataLoader(
    #     RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=30),
    #     batch_size=BSIZE, num_workers=8, shuffle=True)
    # test_loader = DataLoader(
    #     RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=10),
    #     batch_size=BSIZE, num_workers=8)

    current_best = None
    for current_epoch in range(max_epochs):
        data_pass(mdn_rnn, vae, experiment, optimizer, train_dataloader, latent_size, device, current_epoch, train=True)

        test_loss = data_pass(mdn_rnn, vae, experiment, optimizer, test_dataloader, latent_size, device, current_epoch,
                              train=False)

        # scheduler.step(test_loss)
        # earlystopping.step(test_loss)
        if not experiment.debug:
            is_best = not current_best or test_loss < current_best

            if is_best:
                current_best = test_loss

            save_checkpoint({
                "epoch": current_epoch,
                "state_dict": mdn_rnn.state_dict(),
                "optimizer": optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                # 'earlystopping': earlystopping.state_dict(),
                # "precision": test_loss,
            }, is_best, checkpoint_filename, best_model_filename)

        # if earlystopping.stop:
        #     print("End of Training because of early stopping at epoch {}".format(e))
        #     break


if __name__ == "__main__":
    main()
