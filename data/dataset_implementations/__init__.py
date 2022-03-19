from torch.utils.data import DataLoader

from data.dataset_implementations.vae import (
    GUISingleSequenceObservationDataset, GUIMultipleSequencesObservationDataset, GUIEnvImageDataset,
    GUIEnvImageDataset500k, GUIEnvImageDataset300k
)
from data.dataset_implementations.rnn import (
    GUISingleSequenceDataset, GUIMultipleSequencesIdenticalLengthDataset, GUIMultipleSequencesVaryingLengths,
    GUIEnvMultipleSequencesVaryingLengthsIndividualDataLoaders,
    GUIEnvSequencesDatasetRandomWidget500k, GUIEnvSequencesDatasetMixed3600k, GUISequenceBatchSampler,
    GUIEnvSequencesDatasetIndividualDataLoadersRandomWidget500k,
    GUIEnvSequencesDatasetIndividualDataLoadersRandomClicks500k,
    GUIEnvSequencesDatasetIndividualDataLoadersMixed3600k,
    GUIEnvSequencesDatasetIndividualDataLoadersMixed1200k
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
    "multiple_sequences_varying_length_rnn": GUIMultipleSequencesVaryingLengths,
    "multiple_sequences_varying_length_individual_data_loaders_rnn": GUIEnvMultipleSequencesVaryingLengthsIndividualDataLoaders,
    "gui_env_sequences_dataset_random_widget_500k": GUIEnvSequencesDatasetRandomWidget500k,
    "gui_env_sequences_dataset_mixed_3600k": GUIEnvSequencesDatasetMixed3600k,
    "gui_env_sequences_dataset_individual_data_loaders_random_widget_500k": GUIEnvSequencesDatasetIndividualDataLoadersRandomWidget500k,
    "gui_env_sequences_dataset_individual_data_loaders_random_clicks_500k": GUIEnvSequencesDatasetIndividualDataLoadersRandomClicks500k,
    "gui_env_sequences_dataset_individual_data_loaders_mixed_3600k": GUIEnvSequencesDatasetIndividualDataLoadersMixed3600k,
    "gui_env_sequences_dataset_individual_data_loaders_mixed_1200k": GUIEnvSequencesDatasetIndividualDataLoadersMixed1200k
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


def get_main_rnn_data_loader(dataset_name: str, dataset_path: str, split: str, sequence_length: int, batch_size: int,
                             actions_transformation_function, reward_transformation_function,
                             vae_preprocessed_data_path: str, use_shifted_data: bool,
                             shuffle: bool = False, **additional_dataloader_kwargs):

    assert dataset_name in ["gui_env_sequences_dataset_individual_data_loaders_random_widget_500k",
                            "gui_env_sequences_dataset_individual_data_loaders_random_clicks_500k",
                            "gui_env_sequences_dataset_individual_data_loaders_mixed_3600k",
                            "gui_env_sequences_dataset_individual_data_loaders_mixed_1200k",
                            "multiple_sequences_varying_length_individual_data_loaders_rnn"]

    dataset_type = select_rnn_dataset(dataset_name)

    dataset = dataset_type(
        root_dir=dataset_path,
        split=split,
        sequence_length=sequence_length,
        vae_preprocessed_data_path=vae_preprocessed_data_path,
        use_shifted_data=use_shifted_data,
        actions_transformation_function=actions_transformation_function,
        rewards_transformation_function=reward_transformation_function
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

    return dataset, data_loader


def get_individual_rnn_data_loaders(rnn_sequence_dataloader: GUIEnvMultipleSequencesVaryingLengthsIndividualDataLoaders,
                                    batch_size: int, shuffle: bool, **additional_dataloader_kwargs):
    data_loaders = []

    for sequence_idx, sequence in enumerate(rnn_sequence_dataloader):
        sequence_data_loader = DataLoader(
            dataset=sequence,
            batch_size=batch_size,
            shuffle=shuffle,
            **additional_dataloader_kwargs,
            drop_last=True
        )

        data_loaders.append(sequence_data_loader)

    return data_loaders


def get_rnn_dataloader(dataset_name: str, dataset_path: str, split: str, sequence_length: int, batch_size: int,
                       actions_transformation_function, reward_transformation_function,
                       vae_preprocessed_data_path: str, use_shifted_data: bool,
                       shuffle: bool = False, **additional_dataloader_kwargs):
    dataset_type = select_rnn_dataset(dataset_name)

    dataset = dataset_type(
        dataset_path,
        split,
        sequence_length,
        vae_preprocessed_data_path,
        use_shifted_data,
        actions_transformation_function,
        reward_transformation_function
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
