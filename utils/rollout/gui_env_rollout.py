import os
import time

import gym
# noinspection PyUnresolvedReferences
import gym_gui_environments
import numpy as np
import numpy.random
import torch
from PIL import Image

from models import Controller
from utils.misc import load_parameters
from utils.setup_utils import load_yaml_config
from utils.training_utils.training_utils import (
    load_rnn_architecture, load_vae_architecture, vae_transformation_functions
)

POSSIBLE_STOP_MODES = ["time", "iterations"]


class GUIEnvRollout:

    def __init__(self, rnn_dir: str, vae_dir: str, device, stop_mode: str, amount: int = 1000,
                 load_best_rnn: bool = True, load_best_vae: bool = True):
        self.rnn_dir = rnn_dir
        self.vae_dir = vae_dir
        self.device = device
        self.stop_mode = stop_mode
        self.amount = amount

        assert self.stop_mode in POSSIBLE_STOP_MODES

        rnn_config = load_yaml_config(os.path.join(self.rnn_dir, "config.yaml"))

        self.vae, _ = load_vae_architecture(self.vae_dir, device=self.device, load_best=load_best_vae,
                                            load_optimizer=False)
        self.vae.eval()

        self.rnn, _ = load_rnn_architecture(self.rnn_dir, self.vae_dir, device=self.device, batch_size=1,
                                            load_best=load_best_rnn, load_optimizer=False)
        self.rnn.eval()

        vae_config = load_yaml_config(os.path.join(self.vae_dir, "config.yaml"))

        latent_size = vae_config["model_parameters"]["latent_size"]
        hidden_size = rnn_config["model_parameters"]["hidden_size"]
        action_size = rnn_config["model_parameters"]["action_size"]
        self.controller = Controller(latent_size, hidden_size, action_size).to(self.device)

        img_size = vae_config["experiment_parameters"]["img_size"]
        dataset = vae_config["experiment_parameters"]["dataset"]
        output_activation_function = vae_config["model_parameters"]["output_activation_function"]
        self.vae_transformation_functions = vae_transformation_functions(
            img_size=img_size,
            dataset=dataset,
            output_activation_function=output_activation_function
        )

        self.env: gym.Env = gym.make("PySideGUI-v0")

        self.denormalize_actions = lambda x: ((x + 1) * 447) / 2

    def rollout(self, controller_parameters):
        self.rnn.initialize_hidden()
        load_parameters(controller_parameters, self.controller)

        ob = self.env.reset()

        total_reward = 0

        t = 0
        start_time = time.time()

        while True:
            if self.stop_mode == "time":
                if time.time() >= start_time + self.amount:
                    break
            else:
                if t >= self.amount:
                    break

            ob = self.vae_transformation_functions(Image.fromarray(ob)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mu, log_var = self.vae.encode(ob)
                z = self.vae.reparameterize(
                    mu, log_var, self.vae.disable_kld, self.vae.apply_value_range_when_kld_disabled
                ).unsqueeze(0)

                actions = self.controller(z, self.rnn.hidden[0])

                # Updates hidden state internally
                self.rnn(z, actions)

            actions = actions.squeeze()
            actions = self.denormalize_actions(actions).round().int()

            ob, rew, done, info = self.env.step((actions[0], actions[1]))

            # Transform to [0, 1] range
            rew /= 100.0

            total_reward += rew

            # Then maximum code coverage is achieved
            if total_reward >= 1.0:
                break

            t += 1

        return total_reward

    def __del__(self):
        self.env.close()


def main():
    os.chdir("/home/pdeubel/PycharmProjects/world-models-testing/")
    rollout_generator = GUIEnvRollout(
        "/home/pdeubel/PycharmProjects/world-models-testing/logs/mdn-rnn/multiple_sequences_varying_length_rnn/version_0",
        torch.device("cpu")
    )

    shape = rollout_generator.controller.fc.weight.size()
    dim_1, dim_2 = shape[0], shape[1]
    rollout_generator.rollout(np.random.randn((dim_1 * dim_2 + dim_1)))


if __name__ == "__main__":
    main()
