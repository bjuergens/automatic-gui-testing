from typing import Tuple

from torch.utils.data import DataLoader

from data.dataset_implementations.vae import GUISingleSequenceObservationDataset, GUIMultipleSequencesObservationDataset
from data.dataset_implementations.rnn import (
    GUISingleSequenceDataset, GUIMultipleSequencesIdenticalLengthDataset, GUISequenceBatchSampler
)

vae_datasets = {
    "single_sequence_vae": GUISingleSequenceObservationDataset,
    "multiple_sequences_vae": GUIMultipleSequencesObservationDataset
}

rnn_datasets = {
    "multiple_sequences_identical_length_rnn": GUIMultipleSequencesIdenticalLengthDataset
}

POSSIBLE_SPLITS = ["train", "val", "test"]

# Key value is the amount of sequences that were used in the data generation (located in root_dir), and the values is
# a list, which tell the end indices of the sequences that are used for the corresponding split. Therefore, the list has
# always a length of three and because we have [train, val, test] splits. See the first example down below for more
# detail.
IMPLEMENTED_SPLITS = {
    16: [10, 12, 16],  # e.g. 16 sequences, 0-10 for training, 10-12 for validation, 12-16 for testing
    10: [7, 8, 10]  # sequences# : 7 training, 1 validation, 2 testing
}


def get_start_and_end_indices_from_split(number_of_sequences: int, split: str) -> Tuple[int, int]:
    assert number_of_sequences in IMPLEMENTED_SPLITS.keys()

    if split == "train":
        start_index = 0
        end_index = IMPLEMENTED_SPLITS[number_of_sequences][0]
    elif split == "val":
        start_index = IMPLEMENTED_SPLITS[number_of_sequences][0]
        end_index = IMPLEMENTED_SPLITS[number_of_sequences][1]
    elif split == "test":
        start_index = IMPLEMENTED_SPLITS[number_of_sequences][1]
        end_index = IMPLEMENTED_SPLITS[number_of_sequences][2]
    else:
        raise RuntimeError(f"Selected split {split} is not supported")

    return start_index, end_index


def _select_dataset(dataset_name, model_type: str):
    if model_type == "VAE":
        dataset_mapper = vae_datasets
    else:
        dataset_mapper = rnn_datasets

    try:
        selected_dataset = dataset_mapper[dataset_name]
    except KeyError:
        raise RuntimeError(f"Dataset '{dataset_name}' for model type '{model_type}' is not known, check "
                           "data/dataset_implementations/__init__.py for available datasets")

    return selected_dataset


def select_vae_dataset(dataset_name: str):
    return _select_dataset(dataset_name, model_type="VAE")


def select_rnn_dataset(dataset_name: str):
    return _select_dataset(dataset_name, model_type="RNN")


def get_vae_dataloader(dataset_name: str, dataset_path: str, split: str, transformation_functions, batch_size: int,
                       shuffle: bool, **additional_dataloader_kwargs):
    dataset_type = select_vae_dataset(dataset_name)

    dataset = dataset_type(
        dataset_path,
        split,
        transformation_functions
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **additional_dataloader_kwargs
    )

    return dataloader


def get_rnn_dataloader(dataset_name: str, dataset_path: str, split: str, vae_output_file_name: str,
                       sequence_length: int, batch_size: int, **additional_dataloader_kwargs):
    dataset_type = select_rnn_dataset(dataset_name)

    dataset = dataset_type(
        dataset_path,
        split,
        vae_output_file_name,
        sequence_length
    )

    batch_sampler = GUISequenceBatchSampler(
        dataset,
        batch_size=batch_size
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        **additional_dataloader_kwargs
    )

    return dataloader
