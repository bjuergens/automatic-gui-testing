import os

from PIL import Image
from torch.utils.data import Dataset

from data.dataset_implementations.possible_splits import POSSIBLE_SPLITS


class GUIEnvImageDataset(Dataset):

    def __init__(self, root_dir, split: str, transform):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        assert split in POSSIBLE_SPLITS, "Chosen split '{}' is not valid".format(split)

        if self.split == "train":
            train_dir_content = os.listdir(os.path.join(self.root_dir, "train"))
            train_dir_content.sort()
            self.image_paths = [os.path.join(self.root_dir, "train", x) for x in train_dir_content]
        elif self.split == "val":
            val_dir_content = os.listdir(os.path.join(self.root_dir, "val"))
            val_dir_content.sort()
            self.image_paths = [os.path.join(self.root_dir, "val", x) for x in val_dir_content]
        else:
            test_dir_content = os.listdir(os.path.join(self.root_dir, "test"))
            test_dir_content.sort()
            self.image_paths = [os.path.join(self.root_dir, "test", x) for x in test_dir_content]

        self.number_of_images = len(self.image_paths)

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)

        return img
