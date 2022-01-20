import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class GUISingleSequenceDataset(Dataset):

    def __init__(self, root_dir: str, vae_output_file_name: str, sequence_length: int):
        self.root_dir = root_dir
        self.vae_output_file_name = vae_output_file_name
        self.sequence_length = sequence_length

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
