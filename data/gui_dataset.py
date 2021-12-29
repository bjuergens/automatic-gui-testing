import os
from bisect import bisect
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GUIMultipleSequencesDataset(Dataset):

    def __init__(self, root_dir, train: bool, sequence_length: int, transform):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        self.sequence_dirs = []
        for _sequence_dir in os.listdir(self.root_dir):
            if _sequence_dir == "mixed":
                continue

            self.sequence_dirs.append(os.path.join(self.root_dir, _sequence_dir))

        if train:
            self.sequence_dirs = self.sequence_dirs[:-2]
        else:
            self.sequence_dirs = self.sequence_dirs[-2:]

        self.sequence_datasets = [
            GUISequenceDataset(_sq_dir, None, self.sequence_length, self.transform) for _sq_dir in self.sequence_dirs
        ]

        self.dataset_lengths = [_sq_dataset.__len__() for _sq_dataset in self.sequence_datasets]
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

        return dataset[index % self.individual_sequence_length]


class GUISequenceDataset(Dataset):

    def __init__(self, root_dir, train: Optional[bool], sequence_length: int, transform):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        data = np.load(os.path.join(self.root_dir, "data.npz"))
        self.rewards: torch.Tensor = torch.from_numpy(data["rewards"])
        self.actions: torch.Tensor = torch.from_numpy(data["actions"])

        self.observation_files = [
            os.path.join(self.root_dir, "observations", img_file)
            for img_file in os.listdir(os.path.join(self.root_dir, "observations"))
        ]
        self.observation_files.sort()

        if train is not None:
            split_percentage = 0.05
            split_index = round(len(self.rewards) * (1 - split_percentage))
            if train:
                self.observation_files = self.observation_files[:split_index + 1]
                self.rewards = self.rewards[:split_index]
                self.actions = self.actions[:split_index]
            else:
                self.observation_files = self.observation_files[split_index:]
                self.rewards = self.rewards[split_index:]
                self.actions = self.actions[split_index:]

        assert self.rewards.size(0) == self.actions.size(0) == (len(self.observation_files) - 1)
        assert self.__len__() > 0, ("Dataset length is 0 or negative, probably too large sequence length or too few "
                                    "data samples")

    def __len__(self):
        return self.rewards.size(0) - self.sequence_length

    def __getitem__(self, index):
        all_observations = []
        for i in range(self.sequence_length + 1):
            image_file_path = self.observation_files[index + i + 1]
            img = Image.open(image_file_path)

            if self.transform:
                img = self.transform(img)

            all_observations.append(img)

        all_observations = torch.stack(all_observations)
        observations = all_observations[:-1]
        next_observations = all_observations[1:]

        rewards = self.rewards[index:index + self.sequence_length]
        actions = self.actions[index:index + self.sequence_length]

        return observations, next_observations, rewards, actions


class GUIDataset(Dataset):

    def __init__(self, root_dir, split: str = "train", transform=None):

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

        if self.transform:
            img = self.transform(img)

        return img
