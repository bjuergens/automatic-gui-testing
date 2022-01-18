import logging
import os
import shutil
from collections import Counter

import click
import fiftyone as fo
import fiftyone.core.utils as fou
import numpy as np
from PIL import Image
from fiftyone import ViewField


MIXED_FOLDER_NAME = "mixed"
PROCESSED_FOLDER_NAME = "deduplicated-images"


def copy_observations_in_one_folder(root_dir: str, mixed_dir: str):
    try:
        os.makedirs(mixed_dir, exist_ok=False)
    except FileExistsError as e:
        raise RuntimeError("Trying to copy observations to mixed folder but it is already present!") from e

    i = 0
    for sequence_dir in os.listdir(root_dir):
        current_dir = os.path.join(root_dir, sequence_dir)
        observations_dir = os.path.join(current_dir, "observations")

        for file in os.listdir(observations_dir):
            shutil.copy(os.path.join(observations_dir, file), f"{mixed_dir}/{i}-{file}")

        i += 1


def deduplicate_images(dataset):

    for sample in dataset:
        sample["file_hash"] = fou.compute_filehash(sample.filepath)
        sample.save()


def find_duplicates(dataset):
    deduplicate_images(dataset)
    filehash_counts = Counter(sample.file_hash for sample in dataset)
    dup_filehashes = [k for k, v in filehash_counts.items() if v > 1]

    print("Number of duplicate file hashes: %d" % len(dup_filehashes))

    dup_view = (dataset
                # Extract samples with duplicate file hashes
                .match(ViewField("file_hash").is_in(dup_filehashes))
                # Sort by file hash so duplicates will be adjacent
                .sort_by("file_hash")
                )

    print("Number of images that have a duplicate: %d" % len(dup_view))
    print("Number of duplicates: %d" % (len(dup_view) - len(dup_filehashes)))

    return dup_view


def remove_duplicates(root_dir: str, mixed_dir, processed_dir):
    dataset_name = os.path.basename(root_dir)

    try:
        dataset = fo.load_dataset(dataset_name)
        logging.info("Found dataset in database with the same name, loading it. Attention if you changed files on "
                     "disk, these changes will (I think) not be present now")
    except ValueError:
        dataset = fo.Dataset.from_dir(
            mixed_dir,
            fo.types.ImageDirectory,
            name=dataset_name
        )

    dup_view = find_duplicates(dataset)

    print("Length of dataset before: %d" % len(dataset))

    _dup_filehashes = set()
    for sample in dup_view:
        if sample.file_hash not in _dup_filehashes:
            _dup_filehashes.add(sample.file_hash)
            continue

        del dataset[sample.id]

    print("Length of dataset after: %d" % len(dataset))

    # Verify that the dataset no longer contains any duplicates
    print("Number of unique file hashes: %d" % len({s.file_hash for s in dataset}))

    dataset.export(export_dir=processed_dir, dataset_type=fo.types.ImageDirectory)


def _custom_comparison(dataset_path):

    image_files = [os.path.join(dataset_path, x) for x in sorted(os.listdir(dataset_path))]
    images = [np.asarray(Image.open(img_file)) for img_file in image_files]

    duplicates = {i: False for i in range(len(image_files))}
    new_duplicate_path = "/tmp/custom-removed-duplicates"
    os.makedirs(new_duplicate_path, exist_ok=True)

    additionals = []

    for i, first_img in enumerate(images):
        for j, second_img in enumerate(images):
            if not duplicates[j] and i != j:
                if np.array_equal(first_img, second_img):
                    duplicates[j] = True

        if not duplicates[i]:
            additionals.append(i)

    for k, v in duplicates.items():
        if not v:
            file_name = image_files[k].split("/")[-1]
            shutil.copy(image_files[k], f"{new_duplicate_path}/{file_name}")

    for add_ in additionals:
        file_name = image_files[add_].split("/")[-1]
        shutil.copy(image_files[add_], f"{new_duplicate_path}/{file_name}")


@click.command()
@click.option("--root-dir", type=str, required=True,
              help="Root dir of the dataset from which the observations shall be de-duplicated")
def main(root_dir: str):
    root_dir = os.path.normpath(root_dir)
    mixed_dir = f"{root_dir}-{MIXED_FOLDER_NAME}"
    processed_dir = f"{root_dir}-{PROCESSED_FOLDER_NAME}"

    assert os.path.exists(root_dir), f"The provided root_dir '{root_dir}' does not exist"

    copy_observations_in_one_folder(root_dir, mixed_dir)

    remove_duplicates(root_dir, mixed_dir, processed_dir)

    print(f"Removed duplicates and saved them to '{os.path.join(root_dir, PROCESSED_FOLDER_NAME)}'")

    shutil.rmtree(mixed_dir)


if __name__ == "__main__":
    main()
