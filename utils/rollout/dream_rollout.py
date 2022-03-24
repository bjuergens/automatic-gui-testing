import os

import torch

from envs.simulated_gui_env import SimulatedGUIEnv
from models import Controller
from utils.misc import load_parameters
from utils.setup_utils import load_yaml_config


class DreamRollout:

    def __init__(self, rnn_dir: str, vae_dir: str, initial_obs_path: str, max_coordinate_size_for_task: int,
                 temperature: float, device,
                 time_limit: int = 1000, load_best_rnn: bool = True, load_best_vae: bool = True,
                 stop_when_total_reward_exceeded: bool = False, render: bool = False):
        self.rnn_dir = rnn_dir
        self.vae_dir = vae_dir
        self.max_coordinate_size_for_task = max_coordinate_size_for_task
        self.temperature = temperature
        self.device = device
        self.time_limit = time_limit
        self.stop_when_total_reward_exceeded = stop_when_total_reward_exceeded
        self.render = render

        # rnn_dir (and vae_dir inside the rnn config) must point to a valid location, if the path has to be adapted
        # this has to be done before passing it to this constructor
        rnn_config = load_yaml_config(os.path.join(self.rnn_dir, "config.yaml"))
        vae_config = load_yaml_config(os.path.join(self.vae_dir, "config.yaml"))
        if vae_config["model_parameters"]["apply_value_range_when_kld_disabled"]:
            raise RuntimeError(f"VAE used apply_value_range_when_kld_disabled but this is not properly implemented "
                               "in the dream rollout")

        latent_size = vae_config["model_parameters"]["latent_size"]
        hidden_size = rnn_config["model_parameters"]["hidden_size"]
        action_size = rnn_config["model_parameters"]["action_size"]
        self.controller = Controller(latent_size, hidden_size, action_size).to(self.device)

        self.simulated_gui_env = SimulatedGUIEnv(
            rnn_dir=self.rnn_dir,
            vae_dir=self.vae_dir,
            initial_obs_path=initial_obs_path,
            max_coordinate_size_for_task=self.max_coordinate_size_for_task,
            temperature=self.temperature,
            device=self.device,
            load_best_rnn=load_best_rnn,
            load_best_vae=load_best_vae,
            render=self.render
        )

    def rollout(self, controller_parameters):
        load_parameters(controller_parameters, self.controller)

        latent_observation = self.simulated_gui_env.reset()
        total_reward = 0

        for t in range(self.time_limit):
            with torch.no_grad():
                controller_output = self.controller(latent_observation, self.simulated_gui_env.rnn.hidden[0])
                actions = self.controller.predict(controller_output)

            latent_observation, reward, done, info = self.simulated_gui_env.step(actions)

            total_reward += reward

            if self.stop_when_total_reward_exceeded and total_reward >= 1.0:
                break

        # Return minus as the CMA-ES implementation minimizes the objective function
        # noinspection PyUnresolvedReferences
        return -total_reward.cpu().item()
