import os
from typing import Tuple, Union

import cv2
import gym
import h5py
import numpy as np
import torch

from models import BaseVAE
from utils.setup_utils import load_yaml_config
from utils.training_utils.training_utils import (
    load_rnn_architecture, load_vae_architecture, get_rnn_action_transformation_function
)


class SimulatedGUIEnv(gym.Env):

    def __init__(self, rnn_dir: str, vae_dir: str, initial_obs_path: str, max_coordinate_size_for_task: int,
                 temperature: float, device: torch.device, load_best_rnn: bool = True, load_best_vae: bool = True,
                 render: bool = False):
        self.rnn_dir = rnn_dir
        self.vae_dir = vae_dir
        self.max_coordinate_size_for_task = max_coordinate_size_for_task

        # Note that temperature is only used in MDN RNN's (in M models where only a LSTM is used this has no usage)
        self.temperature = torch.tensor(temperature, device=device)
        self.device = device
        self.render_enabled = render

        # Stores mu and log_var not z; we want to sample a new z from these everytime we reset
        # Avoids getting fixated on a particular z
        with h5py.File(initial_obs_path, "r") as f:
            self.initial_mu = torch.from_numpy(f["mu"][:]).to(self.device)
            self.initial_log_var = torch.from_numpy(f["log_var"][:]).to(self.device)

        self.rnn, _ = load_rnn_architecture(self.rnn_dir, self.vae_dir, self.device, batch_size=1,
                                            load_best=load_best_rnn, load_optimizer=False)
        self.rnn.eval()

        self.latent_observation = None  # Populated when doing env.reset()

        self.actions_transformation_function = get_rnn_action_transformation_function(
            max_coordinate_size_for_task=self.max_coordinate_size_for_task,
            reduce_action_coordinate_space_by=self.rnn.reduce_action_coordinate_space_by,
            action_transformation_function_type=self.rnn.action_transformation_function_type
        )

        vae_config = load_yaml_config(os.path.join(self.vae_dir, "config.yaml"))
        self.disable_kld = vae_config["model_parameters"]["disable_kld"]
        self.apply_value_range_when_kld_disabled = vae_config["model_parameters"]["apply_value_range_when_kld_disabled"]

        if self.render_enabled:
            self.vae, _ = load_vae_architecture(self.vae_dir, device=self.device, load_best=load_best_vae,
                                                load_optimizer=False)
            self.vae.eval()

    def step(self, actions: Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]]):
        """
        Expects a tuple of two _integers_ as inputs. Integers can also be in 1D Tensor objects.
        The actions must be in [0, self.max_coordinate_size_for_task].

        Will transform the input (for example [10, 220]) into the appropriate input format for that particular RNN.
        For example it can be that it uses reduced action space and tanh actions then it will be somewhere in the
        range of [-1, 1]
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device)

        actions = self.actions_transformation_function(actions).view(1, 1, -1)

        with torch.no_grad():
            rnn_output = self.rnn(self.latent_observation, actions)
            self.latent_observation, reward = self.rnn.predict(rnn_output, self.latent_observation, self.temperature)

        if isinstance(reward, torch.Tensor):
            reward = reward.item()

        return self.latent_observation, reward, False, {}

    def reset(self):
        self.latent_observation = BaseVAE.reparameterize(
            self.initial_mu, self.initial_log_var, self.disable_kld, self.apply_value_range_when_kld_disabled
        ).unsqueeze(0)

        self.rnn.initialize_hidden()

        return self.latent_observation

    def render(self, mode="human"):
        with torch.no_grad():
            reconstruction = self.vae.denormalize(self.vae.decode(self.latent_observation))

        recon = (reconstruction * 255).int().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imshow("SimulatedGUIEnv", recon)
        cv2.waitKey(0)


def main():
    # Debug stuff
    import os
    from utils.setup_utils import get_depending_model_path, resolve_model_path

    controller_dir = "logs/ai-machine/logs/controller/version_0"
    rnn_dir = get_depending_model_path("controller", controller_dir)
    rnn_dir = resolve_model_path(rnn_dir, model_copied=True, location="ai-machine")

    vae_dir = get_depending_model_path("rnn", rnn_dir)
    vae_dir = resolve_model_path(vae_dir, model_copied=True, location="ai-machine")

    device = torch.device("cpu")

    initial_obs_path = os.path.join(vae_dir, "initial_obs_latent.hdf5")

    env = SimulatedGUIEnv(
        rnn_dir=rnn_dir,
        vae_dir=vae_dir,
        initial_obs_path=initial_obs_path,
        max_coordinate_size_for_task=448,
        device=device,
        load_best_rnn=True,
        render=True
    )

    env.reset()
    env.render()

    reward = 0

    test_actions = torch.randint(0, 447, (100, 2))

    for _ in range(10):
        _, rew, _, _ = env.step((10, 11))
        env.render()

        reward += rew

    print(f"Finished with {reward}")


if __name__ == "__main__":
    main()
