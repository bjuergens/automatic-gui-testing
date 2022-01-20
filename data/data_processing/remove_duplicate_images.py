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
from tqdm import tqdm

PROCESSED_FOLDER_NAME = "deduplicated-images"


def deduplicate_images(dataset):
    print("Calculating image hashes")
    for sample in tqdm(dataset):
        sample["file_hash"] = fou.compute_filehash(sample.filepath)
        sample.save()
    print("Finished calculating image hashes")


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


def remove_duplicates(image_dir: str, processed_dir):
    dataset = fo.Dataset.from_dir(
        image_dir,
        fo.types.ImageDirectory
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
@click.option("-d", "--image-dir", type=str, required=True,
              help="Root dir of the dataset from which the observations shall be de-duplicated")
@click.option("--keep-copy/--no-keep-copy", type=bool, default=True,
              help="Keep the copied files that were created in the 'mixed' directory")
def main(image_dir: str, keep_copy: bool):
    image_dir = os.path.normpath(image_dir)
    processed_dir = f"{image_dir}-{PROCESSED_FOLDER_NAME}"

    remove_duplicates(image_dir, processed_dir)

    print(f"Removed duplicates and saved them to '{processed_dir}'")

    if not keep_copy:
        shutil.rmtree(image_dir)


if __name__ == "__main__":
    main()
