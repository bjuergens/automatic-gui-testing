import logging
import os
from typing import Tuple

import click
import torch
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap, QMouseEvent
from PySide6.QtWidgets import QDialog, QVBoxLayout, QApplication, QLabel
from torchvision import transforms

from envs.simulated_gui_env import SimulatedGUIEnv
from utils.setup_utils import (
    initialize_logger, load_yaml_config, get_device, get_depending_model_path, resolve_model_path
)
from utils.training_utils.training_utils import generate_initial_observation_latent_vector


class DecodedImageLabel(QLabel):

    def __init__(self, simulated_gui_env: SimulatedGUIEnv, **kwargs):
        super().__init__(**kwargs)

        self.simulated_gui_env = simulated_gui_env

        self.to_pil_image_transformation = transforms.ToPILImage()

        observation = self.simulated_gui_env.reset()

        self.reconstruct_and_set_image(observation)

    def reconstruct_and_set_image(self, observation):
        with torch.no_grad():
            reconstructed_image = self.simulated_gui_env.vae.denormalize(
                self.simulated_gui_env.vae.decode(observation.unsqueeze(0))
            )

        self.setPixmap(QPixmap(ImageQt(self.to_pil_image_transformation(reconstructed_image.squeeze(0)))))

    def rollout(self, action: Tuple[int, int]):
        logging.info(f"Rollout on {action[0], action[1]}")

        observation, rew, done, info = self.simulated_gui_env.step(action)
        self.reconstruct_and_set_image(observation)

        logging.info(f"Reward: {rew}")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.rollout(event.position().toTuple())
        super().mouseReleaseEvent(event)


class DreamVisualizationDialog(QDialog):

    def __init__(self, decoded_image_label: DecodedImageLabel, parent=None):
        super().__init__(parent)

        self.decoded_image_label = decoded_image_label
        self.decoded_image_label.setStyleSheet("border: 1px solid black")

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.decoded_image_label)
        self.setLayout(self.layout)


@click.command()
@click.option("-d", "--dir", "rnn_dir", type=str, required=True,
              help="Path to a trained MDN RNN directory")
@click.option("-g", "--gpu", type=int, default=-1, help="Use CPU (-1) or the corresponding GPU to load the models")
@click.option("-t", "--temperature", type=float, default=1.0, help="Temperature parameter of MDNRNN")
@click.option("--best-rnn/--no-best-rnn", type=bool, default=True, help="Load the best RNN or the last checkpoint")
@click.option("--best-vae/--no-best-vae", type=bool, default=True, help="Load the best VAE or the last checkpoint")
@click.option("--vae-copied/--no-vae-copied", type=bool, default=True, help="Was the VAE copied?")
@click.option("--vae-location", type=str, default="local", help="Where was the vae trained (for example ai-machine)?")
def main(rnn_dir: str, gpu: int, temperature: float, best_rnn: bool, best_vae: bool, vae_copied: bool,
         vae_location: str):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    vae_dir = get_depending_model_path("rnn", rnn_dir)
    vae_dir = resolve_model_path(vae_dir, model_copied=vae_copied, location=vae_location)

    device = get_device(gpu)

    initial_obs_path = generate_initial_observation_latent_vector(
            vae_dir=vae_dir,
            device=device,
            load_best=True
    )

    vae_config = load_yaml_config(os.path.join(vae_dir, "config.yaml"))
    img_size = vae_config["experiment_parameters"]["img_size"]

    env = SimulatedGUIEnv(
        rnn_dir=rnn_dir,
        vae_dir=vae_dir,
        initial_obs_path=initial_obs_path,
        max_coordinate_size_for_task=img_size,
        temperature=temperature,
        device=device,
        load_best_rnn=best_rnn,
        load_best_vae=best_vae,
        render=True  # We actually not use the render() function of the env, but when this is True the VAE is loaded
    )

    app = QApplication()
    decoded_image_label = DecodedImageLabel(simulated_gui_env=env)
    dialog = DreamVisualizationDialog(decoded_image_label)
    dialog.show()
    app.exec()


if __name__ == "__main__":
    main()
