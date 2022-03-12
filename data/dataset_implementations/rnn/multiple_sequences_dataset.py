import os
from bisect import bisect

import numpy as np
from torch.utils.data import Dataset

from data.dataset_implementations.possible_splits import get_start_and_end_indices_from_split
from data.dataset_implementations.rnn import GUISingleSequenceDataset, GUISingleSequenceShiftedDataset


class GUIMultipleSequencesIdenticalLengthDataset(Dataset):

    def __init__(self, root_dir, split: str, vae_output_file_name: str, sequence_length: int):
        raise RuntimeError("Current implementation of this dataset is broken, try using the one with varying "
                           "sequence lengths this should also support identical sequence lengths")
        self.root_dir = root_dir
        self.split = split
        self.vae_output_file_name = vae_output_file_name
        self.sequence_length = sequence_length

        self.number_of_sequences = len(os.listdir(self.root_dir))
        self.start_index, self.end_index = get_start_and_end_indices_from_split(self.number_of_sequences, self.split)

        self.sequence_dirs = []

        for sequence_sub_dir in sorted(os.listdir(self.root_dir))[self.start_index:self.end_index]:
            self.sequence_dirs.append(os.path.join(self.root_dir, sequence_sub_dir))

        self.sequence_datasets = [
            GUISingleSequenceDataset(
                seq_dir, self.vae_output_file_name, self.sequence_length
            ) for seq_dir in self.sequence_dirs
        ]

        self.dataset_lengths = [seq_dataset.__len__() for seq_dataset in self.sequence_datasets]
        self.individual_sequence_length = self.dataset_lengths[0]
        self.summed_length = sum(self.dataset_lengths)
        assert all([x == self.individual_sequence_length for x in self.dataset_lengths]), "Sequences are not of the same length"

        self.cumulated_sizes = []
        for i in range(len(self.sequence_datasets)):
            self.cumulated_sizes.append(i * self.individual_sequence_length)

    def __len__(self):
        return self.summed_length

    def __getitem__(self, index):
        dataset_index = bisect(self.cumulated_sizes, index) - 1

        dataset = self.sequence_datasets[dataset_index]

        return dataset[index % self.individual_sequence_length], dataset_index


class GUIMultipleSequencesVaryingLengths(Dataset):
    def __init__(self, root_dir, split: str, sequence_length: int, vae_preprocessed_data_path: str,
                 use_shifted_data: bool, actions_transformation_function=None, rewards_transformation_function=None):
        self.root_dir = root_dir

        assert split in ["train", "val", "test"]
        self.split = split

        self.sequence_length = sequence_length
        self.vae_preprocessed_data_path = vae_preprocessed_data_path
        self.use_shifted_data = use_shifted_data
        self.actions_transformation_function = actions_transformation_function
        self.rewards_transformation_function = rewards_transformation_function

        if self.use_shifted_data:
            single_sequence_dataset_type = GUISingleSequenceShiftedDataset
        else:
            single_sequence_dataset_type = GUISingleSequenceDataset

        self.sequence_datasets = []
        images_dir = os.path.join(self.root_dir, self.split)

        images_dir_content = os.listdir(images_dir)
        images_dir_content.sort()

        for sub_dir_sequence_length in images_dir_content:
            current_sub_dir = os.path.join(images_dir, sub_dir_sequence_length)
            current_sub_dir_content = os.listdir(current_sub_dir)
            current_sub_dir_content.sort()

            for sequence_dir in current_sub_dir_content:
                hdf5_data_group_path = f"/{self.split}/{sub_dir_sequence_length}/{sequence_dir}"
                self.sequence_datasets.append(
                    single_sequence_dataset_type(os.path.join(current_sub_dir, sequence_dir), self.sequence_length,
                                                 self.vae_preprocessed_data_path, hdf5_data_group_path,
                                                 self.actions_transformation_function,
                                                 self.rewards_transformation_function)
                )

        self.lengths_of_sequences = [seq_dataset.__len__() for seq_dataset in self.sequence_datasets]
        self.cumulated_sizes = np.cumsum(np.hstack([0, self.lengths_of_sequences]))

        self.number_of_sequences = len(self.sequence_datasets)

    def get_sequence(self, sequence_index: int) -> GUISingleSequenceShiftedDataset:
        return self.sequence_datasets[sequence_index]

    def get_validation_sequences_for_m_model_comparison(self):
        return NotImplementedError("This method is only implemented for selected datasets used for actual training. "
                                   "Disable the M Model comparison evaluation if you want to use this current dataset")

    def __len__(self):
        return self.number_of_sequences

    def __getitem__(self, index: int):
        sequence_dataset_index = bisect(self.cumulated_sizes, index) - 1
        sequence_dataset = self.sequence_datasets[sequence_dataset_index]

        return sequence_dataset[index - self.cumulated_sizes[sequence_dataset_index]], sequence_dataset_index


class GUIEnvMultipleSequencesVaryingLengthsIndividualDataLoaders(GUIMultipleSequencesVaryingLengths):

    def __len__(self):
        return self.number_of_sequences

    def __getitem__(self, index: int):
        return self.sequence_datasets[index]

    def get_validation_sequences_for_m_model_comparison(self):
        validation_sequences = {
            10: self.sequence_datasets[:10],
            20: self.sequence_datasets[10:15],
            50: self.sequence_datasets[15:17],
        }
        return validation_sequences


class GUIEnvSequencesDatasetRandomWidget500k(GUIMultipleSequencesVaryingLengths):
    def __init__(self, root_dir, split: str, sequence_length: int, vae_preprocessed_data_path: str,
                 use_shifted_data: bool, actions_transformation_function=None, rewards_transformation_function=None):
        super().__init__(root_dir, split, sequence_length, vae_preprocessed_data_path, use_shifted_data,
                         actions_transformation_function, rewards_transformation_function)
        if self.split == "train":
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:70]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[70:110]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[110:140]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[140:160]])

            assert len(self.sequence_datasets) == 160
        else:
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:2]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[2:4]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[4:6]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[6:8]])

            assert len(self.sequence_datasets) == 8

    def get_validation_sequences_for_m_model_comparison(self):
        assert self.split == "val"

        validation_sequences = {
            1000: self.sequence_datasets[:2],
            2000: self.sequence_datasets[2:4],
            5000: self.sequence_datasets[4:6],
            10000: self.sequence_datasets[6:8],
        }
        return validation_sequences


class GUIEnvSequencesDatasetMixed3600k(GUIMultipleSequencesVaryingLengths):
    def __init__(self, root_dir, split: str, sequence_length: int, vae_preprocessed_data_path: str,
                 use_shifted_data: bool, actions_transformation_function=None, rewards_transformation_function=None):
        super().__init__(root_dir, split, sequence_length, vae_preprocessed_data_path, use_shifted_data,
                         actions_transformation_function, rewards_transformation_function)

        if self.split == "train":
            # First Random Widget Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:70]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[70:110]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[110:140]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[140:160]])

            # Random Click Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[160:230]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[230:270]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[270:300]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[300:320]])

            # Second Random Widget Run's
            assert all([seq.rewards.size(0) == 30000 for seq in self.sequence_datasets[320:340]])
            assert all([seq.rewards.size(0) == 40000 for seq in self.sequence_datasets[340:360]])
            assert all([seq.rewards.size(0) == 50000 for seq in self.sequence_datasets[360:380]])

            assert len(self.sequence_datasets) == 380
        else:
            # First Random Widget Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:2]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[2:4]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[4:6]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[6:8]])

            # Random Click Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[8:10]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[10:12]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[12:14]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[14:16]])

            assert len(self.sequence_datasets) == 16

    def get_validation_sequences_for_m_model_comparison(self):
        assert self.split == "val"

        validation_sequences = {
            1000: self.sequence_datasets[:2] + self.sequence_datasets[8:10],
            2000: self.sequence_datasets[2:4] + self.sequence_datasets[10:12],
            5000: self.sequence_datasets[4:6] + self.sequence_datasets[12:14],
            10000: self.sequence_datasets[6:8] + self.sequence_datasets[14:16],
        }
        return validation_sequences


###########################
# Individual Data Loaders
#
# This means that the main dataset class stores sequences and individual dataloaders for each sequence are used.
# Another approach (see above) is to have only one dataloader for all sequences but that can be a bit confusing on how
# sequences are indexed and also may be prone to bugs because it is nested. Nevertheless one dataloader could lead to
# faster training.
###########################

class GUIEnvSequencesDatasetIndividualDataLoadersRandomWidget500k(GUIEnvMultipleSequencesVaryingLengthsIndividualDataLoaders):
    def __init__(self, root_dir, split: str, sequence_length: int, vae_preprocessed_data_path: str,
                 use_shifted_data: bool, actions_transformation_function=None, rewards_transformation_function=None):
        super().__init__(root_dir, split, sequence_length, vae_preprocessed_data_path, use_shifted_data,
                         actions_transformation_function, rewards_transformation_function)
        if self.split == "train":
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:70]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[70:110]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[110:140]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[140:160]])

            assert len(self.sequence_datasets) == 160
        else:
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:2]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[2:4]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[4:6]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[6:8]])

            assert len(self.sequence_datasets) == 8

    def get_validation_sequences_for_m_model_comparison(self):
        assert self.split == "val"

        validation_sequences = {
            1000: self.sequence_datasets[:2],
            2000: self.sequence_datasets[2:4],
            5000: self.sequence_datasets[4:6],
            10000: self.sequence_datasets[6:8],
        }
        return validation_sequences


class GUIEnvSequencesDatasetIndividualDataLoadersRandomClicks500k(GUIEnvMultipleSequencesVaryingLengthsIndividualDataLoaders):
    def __init__(self, root_dir, split: str, sequence_length: int, vae_preprocessed_data_path: str,
                 use_shifted_data: bool, actions_transformation_function=None, rewards_transformation_function=None):
        super().__init__(root_dir, split, sequence_length, vae_preprocessed_data_path, use_shifted_data,
                         actions_transformation_function, rewards_transformation_function)
        if self.split == "train":
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:70]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[70:110]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[110:140]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[140:160]])

            assert len(self.sequence_datasets) == 160
        else:
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:2]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[2:4]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[4:6]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[6:8]])

            assert len(self.sequence_datasets) == 8

    def get_validation_sequences_for_m_model_comparison(self):
        assert self.split == "val"

        validation_sequences = {
            1000: self.sequence_datasets[:2],
            2000: self.sequence_datasets[2:4],
            5000: self.sequence_datasets[4:6],
            10000: self.sequence_datasets[6:8],
        }
        return validation_sequences


class GUIEnvSequencesDatasetIndividualDataLoadersMixed3600k(GUIEnvMultipleSequencesVaryingLengthsIndividualDataLoaders):
    def __init__(self, root_dir, split: str, sequence_length: int, vae_preprocessed_data_path: str,
                 use_shifted_data: bool, actions_transformation_function=None, rewards_transformation_function=None):
        super().__init__(root_dir, split, sequence_length, vae_preprocessed_data_path, use_shifted_data,
                         actions_transformation_function, rewards_transformation_function)

        if self.split == "train":
            # First Random Widget Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:70]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[70:110]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[110:140]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[140:160]])

            # Random Click Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[160:230]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[230:270]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[270:300]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[300:320]])

            # Second Random Widget Run's
            assert all([seq.rewards.size(0) == 30000 for seq in self.sequence_datasets[320:340]])
            assert all([seq.rewards.size(0) == 40000 for seq in self.sequence_datasets[340:360]])
            assert all([seq.rewards.size(0) == 50000 for seq in self.sequence_datasets[360:380]])

            assert len(self.sequence_datasets) == 380
        else:
            # First Random Widget Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[:2]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[2:4]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[4:6]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[6:8]])

            # Random Click Run's
            assert all([seq.rewards.size(0) == 1000 for seq in self.sequence_datasets[8:10]])
            assert all([seq.rewards.size(0) == 2000 for seq in self.sequence_datasets[10:12]])
            assert all([seq.rewards.size(0) == 5000 for seq in self.sequence_datasets[12:14]])
            assert all([seq.rewards.size(0) == 10000 for seq in self.sequence_datasets[14:16]])

            assert len(self.sequence_datasets) == 16

    def get_validation_sequences_for_m_model_comparison(self):
        assert self.split == "val"

        validation_sequences = {
            1000: self.sequence_datasets[:2] + self.sequence_datasets[8:10],
            2000: self.sequence_datasets[2:4] + self.sequence_datasets[10:12],
            5000: self.sequence_datasets[4:6] + self.sequence_datasets[12:14],
            10000: self.sequence_datasets[6:8] + self.sequence_datasets[14:16],
        }
        return validation_sequences
