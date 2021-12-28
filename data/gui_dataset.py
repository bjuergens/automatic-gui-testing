import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GUISequenceDataset(Dataset):

    def __init__(self, root_dir, train: bool, sequence_length: int, transform):
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

        assert self.rewards.size(0) == self.actions.size(0) == (len(self.observation_files) - 1)
        assert self.__len__() > 0, ("Dataset length is 0 or negative, probably too large sequence length or too few "
                                    "data samples")

        split_percentage = 0.05
        split_index = round(len(self.rewards) * (1 - split_percentage))
        if train:
            self.observation_files = self.observation_files[:split_index]
            self.rewards = self.rewards[:split_index]
            self.actions = self.actions[:split_index]
        else:
            self.observation_files = self.observation_files[split_index:]
            self.rewards = self.rewards[split_index:]
            self.actions = self.actions[split_index:]

    def __len__(self):
        return self.rewards.size(0) - self.sequence_length - 1

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
