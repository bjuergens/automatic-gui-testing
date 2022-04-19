import os
from typing import List

import click
import torch
from torchvision.utils import save_image

from utils.setup_utils import get_device, load_yaml_config
from utils.training_utils import load_vae_architecture


def _save_sample(sample: torch.Tensor, vae_name: str):
    save_image(sample[0], f"sample_{vae_name}.png")


@click.command()
@click.option("-d", "--dir", "vae_dirs", type=str, required=True, multiple=True,
              help="Path to a trained VAE directory")
@click.option("-g", "--gpu", type=int, default=-1, help="Use CPU (-1) or the corresponding GPU to load the models")
@click.option("--best-vae/--no-best-vae", type=bool, default=True, help="Load the best VAE or the last checkpoint")
@click.option("--seed", type=int, default=1010, help="Random seed for PyTorch")
def main(vae_dirs: List[str], gpu: int, best_vae: bool, seed: int):
    device = get_device(gpu)

    config = load_yaml_config(os.path.join(vae_dirs[0], "config.yaml"))
    latent_size = config["model_parameters"]["latent_size"]

    torch.manual_seed(seed)
    random_latent_vector = torch.randn((1, latent_size))

    for specific_vae_dir in vae_dirs:
        specific_latent_size = load_yaml_config(
            os.path.join(vae_dirs[0], "config.yaml"))["model_parameters"]["latent_size"]
        assert latent_size == specific_latent_size, "Latent sizes do not match for the given VAEs!"

        vae, _ = load_vae_architecture(specific_vae_dir, device, load_best=best_vae)

        vae_version = specific_vae_dir.split("version_")[-1]
        vae_version = "v_" + vae_version

        with torch.no_grad():
            sample = vae.decode(random_latent_vector)

        _save_sample(sample, vae_version)


if __name__ == "__main__":
    main()
