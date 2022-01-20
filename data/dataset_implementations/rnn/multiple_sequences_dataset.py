import os
from bisect import bisect

from torch.utils.data import Dataset

from data.dataset_implementations import get_start_and_end_indices_from_split
from data.dataset_implementations.rnn import GUISingleSequenceDataset


class GUIMultipleSequencesIdenticalLengthDataset(Dataset):

    def __init__(self, root_dir, split: str, vae_output_file_name: str, sequence_length: int):
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
