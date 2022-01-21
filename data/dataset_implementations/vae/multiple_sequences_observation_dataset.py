import os

from PIL import Image
from torch.utils.data import Dataset

from data.dataset_implementations.possible_splits import POSSIBLE_SPLITS, get_start_and_end_indices_from_split


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