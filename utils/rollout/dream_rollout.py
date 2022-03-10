import os

import h5py
import torch

from models import Controller, BaseVAE
from utils.misc import load_parameters
from utils.setup_utils import load_yaml_config
from utils.training_utils.training_utils import load_rnn_architecture


class DreamRollout:

    def __init__(self, rnn_dir: str, vae_dir: str, initial_obs_path: str, device, time_limit: int = 1000,
                 load_best_rnn: bool = True, stop_when_total_reward_exceeded: bool = False):
        self.rnn_dir = rnn_dir
        self.vae_dir = vae_dir
        self.device = device
        self.time_limit = time_limit
        self.stop_when_total_reward_exceeded = stop_when_total_reward_exceeded

        # rnn_dir (and vae_dir inside the rnn config) must point to a valid location, if the path has to be adapted
        # this has to be done before passing it to this constructor
        rnn_config = load_yaml_config(os.path.join(self.rnn_dir, "config.yaml"))

        # Stores mu and log_var not z; we want to sample a new z from these everytime we reset
        # Avoids getting fixated on a particular z
        with h5py.File(initial_obs_path, "r") as f:
            self.initial_mu = torch.from_numpy(f["mu"][:]).to(self.device)
            self.initial_log_var = torch.from_numpy(f["log_var"][:]).to(self.device)

        self.rnn, _ = load_rnn_architecture(self.rnn_dir, self.vae_dir, self.device, batch_size=1,
                                            load_best=load_best_rnn, load_optimizer=False)
        self.rnn.eval()

        vae_config = load_yaml_config(os.path.join(self.vae_dir, "config.yaml"))
        if vae_config["model_parameters"]["apply_value_range_when_kld_disabled"]:
            raise RuntimeError(f"VAE used apply_value_range_when_kld_disabled but this is not properly implemented "
                               "in the dream rollout")

        latent_size = vae_config["model_parameters"]["latent_size"]
        hidden_size = rnn_config["model_parameters"]["hidden_size"]
        action_size = rnn_config["model_parameters"]["action_size"]
        self.controller = Controller(latent_size, hidden_size, action_size).to(self.device)

    def rollout(self, controller_parameters):
        # TODO this should technically be fixed by using a vae model and then using tanh if
        #  apply_value_range_when_kld_disabled is used
        latent_observation = BaseVAE.reparameterization_trick(self.initial_mu, self.initial_log_var).unsqueeze(0)

        self.rnn.initialize_hidden()
        load_parameters(controller_parameters, self.controller)

        total_reward = 0

        for t in range(self.time_limit):

            with torch.no_grad():
                actions = self.controller(latent_observation, self.rnn.hidden[0])
                rnn_output = self.rnn(latent_observation, actions)
                latent_observation, reward = self.rnn.predict(rnn_output, latent_observation)

            total_reward += reward.squeeze()

            if self.stop_when_total_reward_exceeded and total_reward >= 1.0:
                break

        # Return minus as the CMA-ES implementation minimizes the objective function
        # noinspection PyUnresolvedReferences
        return -total_reward.item()
