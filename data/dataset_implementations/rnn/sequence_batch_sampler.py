import numpy as np
from torch.utils.data import Sampler
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from data.dataset_implementations.rnn import GUIMultipleSequencesIdenticalLengthDataset
from data.dataset_implementations.rnn.multiple_sequences_dataset import GUIMultipleSequencesVaryingLengths


class GUISequenceBatchSampler(Sampler):
    def __init__(self, data_source: GUIMultipleSequencesVaryingLengths, batch_size, shuffle: bool = False):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = True
        self.shuffle = shuffle

        if self.shuffle:
            self.sampler = RandomSampler(self.data_source)
        else:
            self.sampler = SequentialSampler(self.data_source)

        self.sequence_lengths = np.copy(self.data_source.lengths_of_sequences)

        if self.drop_last:
            self.sampler_length = sum(self.sequence_lengths // self.batch_size)
        else:
            self.sampler_length = sum((self.sequence_lengths + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        sequence_order = [idx for idx in self.sampler]

        for sequence_idx in sequence_order:
            if self.batch_size > self.sequence_lengths[sequence_idx]:
                raise RuntimeError(f"Chosen batch_size {self.batch_size} exceeds dataset length, which is "
                                   f"{self.sequence_lengths[sequence_idx]}.")
            offset = self.data_source.cumulated_sizes[sequence_idx]
            sequence_sampler = SequentialSampler(self.data_source.get_sequence(sequence_idx))

            batch = []
            for idx in sequence_sampler:
                batch.append(offset + idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            # It can occur that the last batch has less than sequence_length entries (compare drop_last in standard
            # PyTorch dataloader). We always drop this last batch because the hidden state of the RNN's require the
            # sequence length to have a certain size and it cannot be lower or higher. Therefore, self.drop_last is
            # always True, but kept here for future reference and clarity.
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        return self.sampler_length


class GUISequenceBatchSamplerOld(Sampler):
    def __init__(self, data_source: GUIMultipleSequencesIdenticalLengthDataset, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = True

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
        # It can occur that the last batch has less than sequence_length entries (compare drop_last in standard PyTorch
        # dataloader). We always drop this last batch because the hidden state of the RNN's require the sequence length
        # to have a certain size and it cannot be lower or higher. Therefore, self.drop_last is always True, but kept
        # here for future reference and clarity.
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
