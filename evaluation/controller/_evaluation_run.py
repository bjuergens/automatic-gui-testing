import click
import numpy as np

from evaluation.controller.evaluate_controller import evaluation_options
from utils.misc import flatten_parameters
from utils.rollout.gui_env_rollout import GUIEnvRollout
from utils.setup_utils import get_depending_model_path, get_device
from utils.training_utils.training_utils import construct_controller, load_controller_parameters


@click.command()
@evaluation_options
@click.option("--tmp-file", type=str, required=True, help="File where evaluation result is stored")
def main(controller_directory: str, gpu: int, stop_mode: str, amount: int, tmp_file: str):
    # Allow only local evaluations, meaning models have to be on the same device when evaluating
    # Therefore use directory paths directly
    rnn_dir = get_depending_model_path(model_type="controller", model_dir=controller_directory)
    vae_dir = get_depending_model_path(model_type="rnn", model_dir=rnn_dir)

    device = get_device(gpu)

    rollout_helper = GUIEnvRollout(
        rnn_dir=rnn_dir,
        vae_dir=vae_dir,
        device=device,
        stop_mode=stop_mode,
        amount=amount,
        load_best_rnn=True,
        load_best_vae=True
    )

    controller = construct_controller(rnn_dir, vae_dir)
    controller, _ = load_controller_parameters(controller, controller_directory, device)

    reward = rollout_helper.rollout(flatten_parameters(controller.parameters()))

    np.save(tmp_file, reward)


if __name__ == "__main__":
    main()
