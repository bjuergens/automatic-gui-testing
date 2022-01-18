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
PROCESSED_FOLDER_NAME = "deduplicated_images"


def copy_observations_in_one_folder(root_dir: str):
    mixed_dir = os.path.join(root_dir, MIXED_FOLDER_NAME)

    try:
        os.makedirs(mixed_dir, exist_ok=False)
    except FileExistsError as e:
        raise RuntimeError("Trying to copy observations to mixed folder but it is already present!") from e

    i = 0
    for sequence_dir in os.listdir(root_dir):
        if sequence_dir == MIXED_FOLDER_NAME:
            continue

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


def remove_duplicates(root_dir: str):
    dataset_name = os.path.basename(os.path.normpath(root_dir))
    dataset = fo.Dataset.from_dir(
        os.path.join(root_dir, MIXED_FOLDER_NAME),
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

    export_dir = os.path.join(root_dir, PROCESSED_FOLDER_NAME)
    dataset.export(export_dir=export_dir, dataset_type=fo.types.ImageDirectory)


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
@click.option("--copy-first/--no-copy-first", type=bool, default=True,
              help="Copy observations from the sub folder of each sequence. This step is required for the method to "
                   "work, but can also be omitted if already done.")
def main(root_dir: str, copy_first: bool):

    assert os.path.exists(root_dir), f"The provided root_dir '{root_dir}' does not exist"

    if not copy_first:
        assert os.path.exists(os.path.join(root_dir, MIXED_FOLDER_NAME))
    else:
        copy_observations_in_one_folder(root_dir)

    remove_duplicates(root_dir)

    print(f"Removed duplicates and saved them to '{os.path.join(root_dir, PROCESSED_FOLDER_NAME)}'")


if __name__ == "__main__":
    main()
