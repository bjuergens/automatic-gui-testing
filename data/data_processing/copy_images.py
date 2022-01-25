import os
import shutil

import click
from tqdm import tqdm

MIXED_FOLDER_NAME = "mixed"


def copy_observations_in_one_folder(root_dir: str, mixed_dir: str):
    for sequence_dir in os.listdir(root_dir):
        current_dir = os.path.join(root_dir, sequence_dir)
        observations_dir = os.path.join(current_dir, "observations")

        for file in tqdm(os.listdir(observations_dir)):
            shutil.copy(
                os.path.join(observations_dir, file),
                f"{mixed_dir}/{os.path.basename(root_dir)}-{sequence_dir}-{file}"
            )


@click.command()
@click.option("-d", "--root-dir", type=str, required=True,
              help="Root dir of the dataset from which the observations shall be de-duplicated")
@click.option("--copy-save-dir", type=str,
              help="Can be used to provide a (possibly non-empty) folder where the copies shall be stored")
def main(root_dir: str, copy_save_dir: str):
    root_dir = os.path.normpath(root_dir)

    assert os.path.exists(root_dir), f"The provided root_dir '{root_dir}' does not exist"

    if copy_save_dir is not None:
        mixed_dir = os.path.normpath(copy_save_dir)
    else:
        mixed_dir = f"{root_dir}-{MIXED_FOLDER_NAME}"

        if os.path.exists(mixed_dir):
            print("Warning, the mixed-dir already exists, possibly already containing images from another dataset. "
                  "Consider deleting it or specifying a folder for the copied data by setting '--copy-save-dir'.")

    os.makedirs(mixed_dir, exist_ok=True)
    copy_observations_in_one_folder(root_dir, mixed_dir)


if __name__ == "__main__":
    main()
