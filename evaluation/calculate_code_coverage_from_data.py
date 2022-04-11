import logging
import os

import click
import numpy as np

from utils.setup_utils import initialize_logger


@click.command()
@click.option("-d", "--dir", "dataset_dir", type=str, required=True,
              help="Path to a directory containing one or more generated sequences")
def main(dataset_dir: str):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    sequence_dirs = os.listdir(dataset_dir)
    sequence_dirs.sort()

    achieved_code_coverages = []

    for seq_dir in sequence_dirs:
        sequence_data = np.load(os.path.join(dataset_dir, seq_dir, "data.npz"))
        code_coverage = np.sum(sequence_data["rewards"])

        achieved_code_coverages.append(code_coverage)

    logging.info(f"Calculated code coverages for sequences in '{dataset_dir}'")
    logging.info(f"Mean {np.mean(achieved_code_coverages)} - Stddev {np.std(achieved_code_coverages)} - "
                 f"Max {np.max(achieved_code_coverages)} - Min {np.min(achieved_code_coverages)}")


if __name__ == "__main__":
    main()

