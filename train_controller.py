import logging
import os
import sys
from os import mkdir, unlink, listdir, getpid
from time import sleep

# noinspection PyUnresolvedReferences
import comet_ml  # Needs to be imported __before__ torch
import click
from torch.multiprocessing import Process, Queue
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np

from utils.logging.improved_summary_writer import ImprovedSummaryWriter
from utils.misc import load_parameters
from utils.misc import flatten_parameters
from utils.rollout.dream_rollout import DreamRollout
from utils.setup_utils import (
    load_yaml_config, initialize_logger, pretty_json, save_yaml_config, set_seeds, get_device, get_depending_model_path
)


################################################################################
#                           Thread routines                                    #
################################################################################
def debug_worker_routine(p_queue, r_queue,
                         rnn_dir, vae_dir, time_limit, device, stop_when_total_reward_exceeded):
    """
    Same routine as worker_routine, but used for debugging

    Debugging is difficult with subprocesses running, therefore this function can be used without subprocesses.
    """
    with torch.no_grad():
        r_gen = DreamRollout(rnn_dir, vae_dir, device, time_limit, load_best_rnn=True, load_best_vae=True,
                             stop_when_total_reward_exceeded=stop_when_total_reward_exceeded)
        empty_counter = 0
        while True:
            if p_queue.empty():
                sleep(.1)
                empty_counter += 1

                if empty_counter > 20:
                    break
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))
                empty_counter = 0


def worker_routine(p_queue, r_queue, e_queue,
                   tmp_dir, rnn_dir, vae_dir, time_limit, device, stop_when_total_reward_exceeded):
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.

    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    if tmp_dir is not None:
        # redirect streams
        sys.stdout = open(os.path.join(tmp_dir, str(getpid()) + '.out'), 'a')
        sys.stderr = open(os.path.join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = DreamRollout(rnn_dir, vae_dir, device, time_limit, load_best_rnn=True, load_best_vae=True,
                             stop_when_total_reward_exceeded=stop_when_total_reward_exceeded)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(p_queue, r_queue, rnn_dir, vae_dir, time_limit,
             solutions, results, rollouts, device, stop_when_total_reward_exceeded, debug=False):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    if debug:
        debug_worker_routine(p_queue, r_queue, rnn_dir, vae_dir, time_limit, device, stop_when_total_reward_exceeded)

    logging.info("Evaluating...")
    for _ in tqdm(range(rollouts)):
        while r_queue.empty():
            sleep(.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)


@click.command()
@click.option("-c", "--config", "config_path", type=str, required=True,
              help="Path to a YAML configuration containing training options")
@click.option("-l", "--load", "load_path", type=str,
              help=("Path to a previous training, from which training shall continue (will create a new experiment "
                    "directory)"))
@click.option("--disable-comet/--no-disable-comet", type=bool, default=False,
              help="Disable logging to Comet (automatically disabled when API key is not provided in home folder)")
def main(config_path: str, load_path: str, disable_comet: bool):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    config = load_yaml_config(config_path)

    population_size = config["experiment_parameters"]["population_size"]
    sigma = config["experiment_parameters"]["sigma"]
    number_of_samples = config["experiment_parameters"]["number_of_samples"]
    number_of_evaluations = config["experiment_parameters"]["number_of_evaluations"]
    target_return = config["experiment_parameters"]["target_return"]
    stop_when_total_reward_exceeded = config["experiment_parameters"]["stop_when_total_reward_exceeded"]
    time_limit = config["experiment_parameters"]["time_limit"]
    max_generations = config["experiment_parameters"]["max_generations"]

    rnn_dir = config["rnn_parameters"]["rnn_dir"]
    # Use rnn_dir directly, we only want training on local models anyway
    vae_dir = get_depending_model_path(model_type="rnn", model_dir=rnn_dir)

    number_of_workers = config["trainer_parameters"]["num_workers"]
    gpu_id = config["trainer_parameters"]["gpu"]

    debug = config["logging_parameters"]["debug"]
    save_dir = config["logging_parameters"]["save_dir"]
    scalar_log_frequency = config["logging_parameters"]["scalar_log_frequency"]
    save_model_checkpoints = config["logging_parameters"]["save_model_checkpoints"]
    display_progress_bars = config["logging_parameters"]["display_progress_bars"]

    assert max_generations > 0, f"Maximum number of generations must be greater than 0"

    manual_seed = config["experiment_parameters"]["manual_seed"]
    set_seeds(manual_seed)

    device = get_device(gpu_id)

    if not debug:
        assert number_of_workers > 0, f"Number of workers must be greater than 0"

        summary_writer = ImprovedSummaryWriter(
            log_dir=save_dir,
            comet_config={
                "project_name": "world-models/controller",
                "disabled": disable_comet
            }
        )

        # Log hyperparameters to the tensorboard
        summary_writer.add_text("Hyperparameters", pretty_json(config), global_step=0)

        # Unfortunately tensorboardX does not expose this functionality and name cannot be set in constructor
        if not disable_comet:
            # noinspection PyProtectedMember
            summary_writer._get_comet_logger()._experiment.set_name(f"version_{summary_writer.version_number}")

        log_dir = summary_writer.get_logdir()
        best_model_filename = os.path.join(log_dir, "best.pt")

        save_yaml_config(os.path.join(log_dir, "config.yaml"), config)

        # Create tmp dir if non existent and clean it if existent
        tmp_dir = os.path.join(log_dir, "tmp")
        if not os.path.exists(tmp_dir):
            mkdir(tmp_dir)
        else:
            for fname in listdir(tmp_dir):
                unlink(os.path.join(tmp_dir, fname))

        logging.info(f"Started Controller training version_{summary_writer.version_number} for {max_generations} "
                     "generations")
    else:
        summary_writer = None
        log_dir = None
        tmp_dir = None

    ################################################################################
    #                Define queues and start workers                               #
    ################################################################################
    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    if not debug:
        for p_index in range(number_of_workers):
            Process(target=worker_routine, args=(p_queue, r_queue, e_queue,
                                                 tmp_dir, rnn_dir, vae_dir, time_limit, device,
                                                 stop_when_total_reward_exceeded)).start()

    ################################################################################
    #                           Launch CMA                                         #
    ################################################################################
    rnn_config = load_yaml_config(os.path.join(rnn_dir, "config.yaml"))
    vae_dir = rnn_config["vae_parameters"]["directory"]
    vae_config = load_yaml_config(os.path.join(vae_dir, "config.yaml"))
    latent_size = vae_config["model_parameters"]["latent_size"]
    hidden_size = rnn_config["model_parameters"]["hidden_size"]
    action_size = rnn_config["model_parameters"]["action_size"]

    controller = Controller(latent_size, hidden_size, action_size)  # dummy instance

    # Define current best and load parameters
    current_best = None

    if load_path is not None:
        state = torch.load(os.path.join(load_path, "best.pt"), map_location=device)
        # Take minus of the reward because when saving we "convert" it back to the normal way of summing up the fitness
        # For training we however take the negative amount as the CMA-ES implementation minimizes the fitness instead
        # of maximizing it
        current_best = -state["reward"]
        controller.load_state_dict(state["state_dict"])

        if not debug:
            old_config = load_yaml_config(os.path.join(load_path, "config.yaml"))
            old_config["original_location"] = load_path
            save_yaml_config(os.path.join(log_dir, "loaded_from_this_config.yaml"), old_config)

        logging.info(f"Loading previous training from {load_path}. Starting training with newly given configuration")

    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(
        flatten_parameters(parameters),
        sigma,
        {"popsize": population_size, "seed": manual_seed}
    )

    generation = 0

    while not es.stop() and generation < max_generations:

        if current_best is not None and target_return is not None and -current_best > target_return:
            logging.info("Training already better than target return, stopping.")
            break

        r_list = [0] * population_size  # Result list
        solutions = es.ask()

        # Push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(number_of_samples):
                p_queue.put((s_id, s))

        if debug:
            debug_worker_routine(p_queue, r_queue, rnn_dir, vae_dir, time_limit, device, stop_when_total_reward_exceeded)

        if display_progress_bars:
            progress_bar = tqdm(total=population_size * number_of_samples, desc=f"Generation {generation} - Rewards")

        # Take results from result queue
        for _ in range(population_size * number_of_samples):
            while r_queue.empty():
                sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / number_of_samples

            if display_progress_bars:
                progress_bar.update(1)

        if display_progress_bars:
            progress_bar.close()

        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if generation % scalar_log_frequency == 0 or generation == max_generations - 1:
            best_params, best, std_best = evaluate(p_queue, r_queue, rnn_dir, vae_dir, time_limit, solutions, r_list,
                                                   rollouts=number_of_evaluations, device=device,
                                                   stop_when_total_reward_exceeded=stop_when_total_reward_exceeded,
                                                   debug=debug)
            logging.info(f"Current evaluation: {-best}")

            if not debug:
                # Rewards are multiplied with (-1), therefore taking the max and then multiplying with (-1) gives the
                # correct minimum reward for example
                summary_writer.add_scalar("min", -np.max(r_list), global_step=generation)
                summary_writer.add_scalar("max", -np.min(r_list), global_step=generation)
                summary_writer.add_scalar("mean", -np.mean(r_list), global_step=generation)
                summary_writer.add_scalar("best", -best, global_step=generation)

                if save_model_checkpoints:
                    # noinspection PyUnboundLocalVariable
                    if not os.path.exists(best_model_filename) or current_best is None or best < current_best:
                        current_best = best
                        logging.info(f"Saving new best with value {-current_best}+-{std_best}...")
                        load_parameters(best_params, controller)

                        # noinspection PyUnboundLocalVariable
                        torch.save(
                            {"generation": generation,
                             "reward": -current_best,
                             "state_dict": controller.state_dict()},
                            best_model_filename)

            if target_return is not None and -best > target_return:
                logging.info(f"Terminating controller training with achieved value {-best}")
                break

        generation += 1

    es.result_pretty()
    e_queue.put("EOP")

    if not debug:
        # Use prefix e for experiment_parameters to avoid possible reassignment of a hparam when combining with
        # other parameters
        exp_params = {f"e_{k}": v for k, v in config["experiment_parameters"].items()}
        rnn_params = {f"rnn_{k}": v for k, v in config["rnn_parameters"].items()}
        trainer_params = {f"t_{k}": v for k, v in config["trainer_parameters"].items()}
        logging_params = {f"l_{k}": v for k, v in config["logging_parameters"].items()}

        hparams = {**exp_params, **rnn_params, **trainer_params, **logging_params}

        if current_best is None:
            current_best = 0

        summary_writer.add_hparams(
            hparams,
            {"hparams/best": -current_best},
            name="hparams"
        )
        # Ensure everything is logged to the tensorboard
        summary_writer.flush()


if __name__ == "__main__":
    # To share CUDA tensors between subprocesses we have to use "spawn" as the starting method for the subprocesses
    # Compare also with https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    torch.multiprocessing.set_start_method("spawn")
    main()
