from torch.utils.data import DataLoader

from data.dataset_implementations.vae import (
    GUISingleSequenceObservationDataset, GUIMultipleSequencesObservationDataset, GUIEnvImageDataset,
    GUIEnvImageDataset500k, GUIEnvImageDataset300k
)
from data.dataset_implementations.rnn import (
    GUISingleSequenceDataset, GUIMultipleSequencesIdenticalLengthDataset, GUIEnvSequencesDatasetRandomWidget500k,
    GUISequenceBatchSampler
)

vae_datasets = {
    "single_sequence_vae": GUISingleSequenceObservationDataset,
    "multiple_sequences_vae": GUIMultipleSequencesObservationDataset,
    "gui_env_image_dataset": GUIEnvImageDataset,
    "gui_env_image_dataset_500k_normalize": GUIEnvImageDataset500k,
    "gui_env_image_dataset_300k": GUIEnvImageDataset300k
}

rnn_datasets = {
    "multiple_sequences_identical_length_rnn": GUIMultipleSequencesIdenticalLengthDataset,
    "gui_env_sequences_dataset_random_widget_500k": GUIEnvSequencesDatasetRandomWidget500k
}


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


def get_rnn_dataloader(dataset_name: str, dataset_path: str, split: str,
                       sequence_length: int, batch_size: int, vae_preprocessed_data_path: str, use_shifted_data: bool,
                       shuffle: bool = False, **additional_dataloader_kwargs):
    dataset_type = select_rnn_dataset(dataset_name)

    dataset = dataset_type(
        dataset_path,
        split,
        sequence_length,
        vae_preprocessed_data_path,
        use_shifted_data
    )

    batch_sampler = GUISequenceBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        **additional_dataloader_kwargs
    )

    return dataloader
