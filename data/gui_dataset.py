import os
from bisect import bisect
from typing import Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import SequentialSampler

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


class GUIMultipleSequencesObservationDataset(Dataset):

    def __init__(self, root_dir, split: str, transform):
        self.root_dir = root_dir
        self.transform = transform

        assert split in POSSIBLE_SPLITS
        self.split = split

        self.number_of_sequences = len(os.listdir(self.root_dir))
        self.start_index, self.end_index = get_start_and_end_indices_from_split(self.number_of_sequences, self.split)

        self.observation_images = []

        for sequence_sub_dir in sorted(os.listdir(self.root_dir))[self.start_index:self.end_index]:
            sequence_images_list = sorted(os.listdir(os.path.join(self.root_dir, sequence_sub_dir, "observations")))

            self.observation_images.extend(
                [os.path.join(self.root_dir, sequence_sub_dir, "observations", x) for x in sequence_images_list]
            )

        self.number_of_observations = len(self.observation_images)

    def __len__(self):
        return self.number_of_observations

    def __getitem__(self, index):
        return self.transform(Image.open(self.observation_images[index]))


class GUIMultipleSequencesDataset(Dataset):

    def __init__(self, root_dir, split: str, vae_output_file_name: str, sequence_length: int, transform):
        self.root_dir = root_dir
        self.split = split
        self.vae_output_file_name = vae_output_file_name
        self.sequence_length = sequence_length
        self.transform = transform

        self.number_of_sequences = len(os.listdir(self.root_dir))
        self.start_index, self.end_index = get_start_and_end_indices_from_split(self.number_of_sequences, self.split)

        self.sequence_dirs = []

        for sequence_sub_dir in sorted(os.listdir(self.root_dir))[self.start_index:self.end_index]:
            self.sequence_dirs.append(os.path.join(self.root_dir, sequence_sub_dir))

        self.sequence_datasets = [
            GUISequenceDataset(
                seq_dir, vae_output_file_name, self.sequence_length, self.transform
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


class GUISequenceBatchSampler(Sampler):
    def __init__(self, data_source: GUIMultipleSequencesDataset, batch_size, drop_last=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.sampler = SequentialSampler(self.data_source)

        self.max_sequence_index = self.data_source.dataset_lengths[0]

    def __iter__(self):
        batch = []
        current_stop_point = self.max_sequence_index
        for idx in self.sampler:
            if idx >= current_stop_point:
                if len(batch) == self.batch_size:
                    yield batch
                batch = []
                current_stop_point += self.max_sequence_index
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class GUISequenceDataset(Dataset):

    def __init__(self, root_dir: str, vae_output_file_name: str, sequence_length: int, transform):
        self.root_dir = root_dir
        self.vae_output_file_name = vae_output_file_name
        self.sequence_length = sequence_length
        self.transform = transform

        with np.load(os.path.join(self.root_dir, "data.npz")) as data:
            self.rewards: torch.Tensor = torch.from_numpy(data["rewards"])
            self.actions: torch.Tensor = torch.from_numpy(data["actions"])

        self.vae_preprocessed_data = h5py.File(os.path.join(self.root_dir, self.vae_output_file_name), "r")

        self.mus = self.vae_preprocessed_data["mus"]
        self.log_vars = self.vae_preprocessed_data["log_vars"]

        assert self.rewards.size(0) == self.actions.size(0) == (self.mus.shape[0] - 1) == (self.log_vars.shape[0] - 1)
        assert self.__len__() > 0, ("Dataset length is 0 or negative, probably too large sequence length or too few "
                                    "data samples")

    def __len__(self):
        return self.rewards.size(0) - self.sequence_length

    def __getitem__(self, index):
        sub_sequence_mus = self.mus[index:index + self.sequence_length + 1]
        sub_sequence_log_vars = self.log_vars[index:index + self.sequence_length + 1]

        mus = sub_sequence_mus[:-1]
        next_mus = sub_sequence_mus[1:]
        log_vars = sub_sequence_log_vars[:-1]
        next_log_vars = sub_sequence_log_vars[1:]

        rewards = self.rewards[index:index + self.sequence_length]
        actions = self.actions[index:index + self.sequence_length]

        return mus, next_mus, log_vars, next_log_vars, rewards, actions


class GUIDataset(Dataset):

    def __init__(self, root_dir, split: str, transform):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.train_split = 0.65
        self.val_split = 0.15
        self.test_split = 0.2

        possible_splits = [
            "train",
            "val",
            "test"
        ]

        assert split in possible_splits, "Chosen split '{}' is not valid".format(split)

        self.train_index = round(len(os.listdir(self.root_dir)) * self.train_split)
        self.val_index = round(len(os.listdir(self.root_dir)) * self.val_split)

        self.train_length = len(os.listdir(self.root_dir)[:self.train_index])
        self.val_length = len(os.listdir(self.root_dir)[self.train_index:self.train_index + self.val_index])
        self.test_length = len(os.listdir(self.root_dir)[self.train_index + self.val_index:])

    def __len__(self):
        if self.split == "train":
            return self.train_length
        elif self.split == "val":
            return self.val_length
        else:
            return self.test_length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            raise RuntimeError("Index is a Torch.Tensor object which is not supported in the GUIDataset")

        if self.split == "train":
            image_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[index])
        elif self.split == "val":
            image_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[self.train_index + index])
        else:
            image_path = os.path.join(
                self.root_dir,
                os.listdir(self.root_dir)[self.train_index + self.val_index + index]
            )

        img = Image.open(image_path)
        img = self.transform(img)

        return img
