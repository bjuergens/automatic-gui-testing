import logging
import math
import os

import click
import cv2
import torch
import yaml
from PIL import Image
from torchvision.transforms import transforms

from models import VAE, MDRNN
from utils.misc import initialize_logger


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Event {event} on {x, y}")


def rollout(mdn_rnn, vae, transformation_functions):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse)

    initial_obs_file_path = "datasets/gui_env/random-widgets/2021-12-29_19-02-29/0/observations/00000000.png"
    initial_image = Image.open(initial_obs_file_path)

    initial_image = transformation_functions(initial_image)

    with torch.no_grad():
        _, mu, log_var = vae(initial_image.unsqueeze(0))

    sigma = torch.exp(0.5 * log_var)
    eps = torch.randn_like(sigma)

    z = eps.mul(sigma).add_(mu)
    with torch.no_grad():
        recon_initial_image = vae.decoder(z)

    cv2.imshow("image", recon_initial_image[0].numpy().transpose(1, 2, 0))
    z = z.unsqueeze(0)

    while True:
        action = torch.randint(low=0, high=224, size=(2,)).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            mus, sigmas, log_pi, rewards = mdn_rnn(action, z)
            z = mdn_rnn.predict_next_latent(z, mus=mus, sigmas=sigmas, log_pi=log_pi)

            reconstruction = vae.decoder(z)

        cv2.imshow("image", reconstruction[0].numpy().transpose(1, 2, 0))

        key = cv2.waitKey()

        if key == ord("q"):
            break
        elif key == ord("c"):
            continue


@click.command()
@click.option("-d", "--dir", "dir_path", type=str, required=True,
              help="Path to a trained MDN RNN directory")
def main(dir_path):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    config_path = os.path.join(dir_path, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if config["trainer_parameters"]["gpu"] >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['trainer_parameters']['gpu']}")
    else:
        device = torch.device("cpu")

    # vae_directory = os.path.join("/home/pdeubel/desktop", config["vae_parameters"]["directory"])
    vae_directory = config["vae_parameters"]["directory"]

    with open(os.path.join(vae_directory, "config.yaml")) as vae_config_file:
        vae_config = yaml.safe_load(vae_config_file)

    latent_size = vae_config["model_parameters"]["latent_size"]

    vae_name = vae_config["model_parameters"]["name"]

    vae = VAE(3, latent_size).to(device)
    checkpoint = torch.load(os.path.join(vae_directory, "best.pt"), map_location=device)
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()

    action_size = 2
    hidden_size = config["model_parameters"]["hidden_size"]
    mdn_rnn = MDRNN(latent_size, action_size, hidden_size, gaussians=5, batch_size=1, device=device).to(device)
    mdn_rnn_checkpoint = torch.load(os.path.join(dir_path, "best.pt"), map_location=device)
    mdn_rnn.load_state_dict(mdn_rnn_checkpoint["state_dict"])
    mdn_rnn.eval()

    set_range = transforms.Lambda(lambda x: 2 * x - 1.0)

    transformation_functions = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        set_range
    ])

    rollout(mdn_rnn, vae, transformation_functions)


if __name__ == "__main__":
    main()
