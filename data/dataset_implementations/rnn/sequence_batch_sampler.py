from torch.utils.data import Sampler
from torch.utils.data.sampler import SequentialSampler

from data.dataset_implementations.rnn import GUIMultipleSequencesIdenticalLengthDataset


class GUISequenceBatchSampler(Sampler):
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
