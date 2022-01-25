import os

from PIL import Image
from torch.utils.data import Dataset

from data.dataset_implementations.possible_splits import POSSIBLE_SPLITS


class GUISingleSequenceObservationDataset(Dataset):

    def __init__(self, root_dir, split: str, transform):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.train_split = 0.65
        self.val_split = 0.15
        self.test_split = 0.2

        assert split in POSSIBLE_SPLITS, "Chosen split '{}' is not valid".format(split)

        root_dir_content = os.listdir(self.root_dir)
        root_dir_content.sort()

        root_dir_len = len(root_dir_content)

        self.train_index = round(root_dir_len * self.train_split)
        self.val_index = round(root_dir_len * self.val_split)

        if self.split == "train":
            self.image_paths = [os.path.join(self.root_dir, x) for x in root_dir_content[:self.train_index]]
        elif self.split == "val":
            self.image_paths = [os.path.join(self.root_dir, x) for x in root_dir_content[self.train_index:self.train_index + self.val_index]]
        else:
            self.image_paths = [os.path.join(self.root_dir, x) for x in root_dir_content[self.train_index + self.val_index:]]

        self.number_of_images = len(self.image_paths)

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)

        return img
