import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class GUISingleSequenceDataset(Dataset):

    def __init__(self, root_dir: str, sequence_length: int, vae_preprocessed_data_path: str, hdf5_data_group_path: str):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.vae_preprocessed_data_path = vae_preprocessed_data_path
        self.hdf5_data_group_path = hdf5_data_group_path

        with np.load(os.path.join(self.root_dir, "data.npz")) as data:
            self.rewards: torch.Tensor = torch.from_numpy(data["rewards"])
            self.actions: torch.Tensor = torch.from_numpy(data["actions"])

        self.vae_preprocessed_data = h5py.File(vae_preprocessed_data_path, "r")

        self.mus = self.vae_preprocessed_data[f"{self.hdf5_data_group_path}/mus"]
        self.log_vars = self.vae_preprocessed_data[f"{self.hdf5_data_group_path}/log_vars"]

        self.dataset_length = self.rewards.size(0) - self.sequence_length

        assert self.rewards.size(0) == self.actions.size(0) == (self.mus.shape[0] - 1) == (self.log_vars.shape[0] - 1)
        assert self.__len__() > 0, ("Dataset length is 0 or negative, probably too large sequence length or too few "
                                    "data samples")

    def __len__(self):
        return self.dataset_length

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


class GUISingleSequenceShiftedDataset(Dataset):

    def __init__(self, root_dir: str, sequence_length: int, vae_preprocessed_data_path: str, hdf5_data_group_path: str):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.vae_preprocessed_data_path = vae_preprocessed_data_path
        self.hdf5_data_group_path = hdf5_data_group_path

        with np.load(os.path.join(self.root_dir, "data.npz")) as data:
            self.rewards: torch.Tensor = torch.from_numpy(data["rewards"])
            self.actions: torch.Tensor = torch.from_numpy(data["actions"])

        self.vae_preprocessed_data = h5py.File(vae_preprocessed_data_path, "r")

        self.mus = self.vae_preprocessed_data[f"{self.hdf5_data_group_path}/mus"]
        self.log_vars = self.vae_preprocessed_data[f"{self.hdf5_data_group_path}/log_vars"]

        self.dataset_length = self.rewards.size(0) // self.sequence_length

        assert self.rewards.size(0) == self.actions.size(0) == (self.mus.shape[0] - 1) == (self.log_vars.shape[0] - 1)
        assert self.__len__() > 0, ("Dataset length is 0 or negative, probably too large sequence length or too few "
                                    "data samples")

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        index_start_point = index * self.sequence_length
        sub_sequence_mus = self.mus[index_start_point:index_start_point + self.sequence_length + 1]
        sub_sequence_log_vars = self.log_vars[index_start_point:index_start_point + self.sequence_length + 1]

        mus = sub_sequence_mus[:-1]
        next_mus = sub_sequence_mus[1:]
        log_vars = sub_sequence_log_vars[:-1]
        next_log_vars = sub_sequence_log_vars[1:]

        rewards = self.rewards[index_start_point:index_start_point + self.sequence_length]
        actions = self.actions[index_start_point:index_start_point + self.sequence_length]

        return mus, next_mus, log_vars, next_log_vars, rewards, actions
