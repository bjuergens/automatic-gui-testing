import logging
import os
import random
import shutil
from copy import deepcopy

import click
from tqdm import tqdm

from utils.setup_utils import initialize_logger

SPLIT_FOLDER_NAME = "splits"


@click.command()
@click.option("-d", "--root-dir", type=str, required=True,
              help="Path to a directory with splits. Tests if there are duplicates.")
def test_if_duplicates(root_dir: str):
    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    train_files = os.listdir(os.path.join(root_dir, "train"))
    val_files = os.listdir(os.path.join(root_dir, "val"))
    test_files = os.listdir(os.path.join(root_dir, "test"))

    all_files = deepcopy(train_files)
    all_files.extend(val_files)
    all_files.extend(test_files)

    all_files_set = set(all_files)

    assert len(train_files) + len(val_files) + len(test_files) == len(all_files_set)

    logging.info("The directory containing the splits has no duplicates")


@click.command()
@click.option("-d", "--root-dir", type=str, required=True,
              help="Path to a directory containing only images. Will create a new folder containing the train/val/test "
                   "splits.")
def main(root_dir: str):
    random.seed(1010)

    logger, _ = initialize_logger()
    logger.setLevel(logging.INFO)

    logging.info("Creating dataset splits")

    original_dataset_length = len(os.listdir(root_dir))

    splits_dir = f"{root_dir}-{SPLIT_FOLDER_NAME}"

    if os.path.exists(splits_dir):
        raise RuntimeError(f"Splits directory already exists ({splits_dir}")
    os.makedirs(splits_dir)

    train_dir = os.path.join(splits_dir, "train")
    val_dir = os.path.join(splits_dir, "val")
    test_dir = os.path.join(splits_dir, "test")
    all_images_dir = os.path.join(splits_dir, "all_images")
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)
    os.makedirs(all_images_dir)

    logging.info(f"Copying image files to new directory '{all_images_dir}'")
    for file_path in tqdm(os.listdir(root_dir)):
        shutil.copy(os.path.join(root_dir, file_path), os.path.join(all_images_dir, file_path))
    logging.info("Finished copying image files")

    splits_dir_content = os.listdir(all_images_dir)

    train_split_percentage = 0.98
    val_split_percentage = 0.01
    test_split_percentage = 0.01

    number_of_train_images = round(len(splits_dir_content) * train_split_percentage)
    number_of_val_images = round(len(splits_dir_content) * val_split_percentage)
    number_of_test_images = len(splits_dir_content) - number_of_train_images - number_of_val_images

    train_images = random.sample(splits_dir_content, number_of_train_images)

    logging.info("Moving train images")
    for img_file in tqdm(train_images):
        shutil.move(os.path.join(all_images_dir, img_file), os.path.join(train_dir, img_file))

    splits_dir_content = os.listdir(all_images_dir)

    val_images = random.sample(splits_dir_content, number_of_val_images)

    logging.info("Moving validation images")
    for img_file in tqdm(val_images):
        shutil.move(os.path.join(all_images_dir, img_file), os.path.join(val_dir, img_file))

    splits_dir_content = os.listdir(all_images_dir)

    assert len(splits_dir_content) == number_of_test_images

    logging.info("Moving test images")
    for img_file in tqdm(splits_dir_content):
        shutil.move(os.path.join(all_images_dir, img_file), os.path.join(test_dir, img_file))

    shutil.rmtree(all_images_dir)

    new_train_length = len(os.listdir(train_dir))
    new_val_length = len(os.listdir(val_dir))
    new_test_length = len(os.listdir(test_dir))

    assert new_train_length + new_val_length + new_test_length == original_dataset_length

    logging.info(f"Finished creating splits in directory ''{splits_dir}")


if __name__ == "__main__":
    main()
    # test_if_duplicates()
