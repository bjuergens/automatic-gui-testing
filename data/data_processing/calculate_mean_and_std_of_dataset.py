import click
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data.dataset_implementations import GUIEnvImageDataset


@click.command()
@click.option("-d", "--root-dir", type=str, required=True,
              help="Root dir of the dataset, only GUIEnvImageDataset is supported!")
def main(root_dir: str):

    dataset = GUIEnvImageDataset(
        root_dir=root_dir,
        split="train",
        transform=transforms.ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    sum_per_channel = torch.zeros(3, dtype=torch.float64)
    squared_sum_per_channel = torch.zeros(3, dtype=torch.float64)
    number_of_data_points = 0

    for batch_id, data in tqdm(enumerate(dataloader)):
        sum_per_channel += data.sum((0, 2, 3))
        squared_sum_per_channel += data.pow(2).sum((0, 2, 3))
        number_of_data_points += (data.size(0) * data.size(2) * data.size(3))

        if batch_id % 250 == 0:
            print(f"Current sum_per_channel: {sum_per_channel}")

    dataset_mean = sum_per_channel / number_of_data_points
    dataset_stddev = torch.sqrt((squared_sum_per_channel / number_of_data_points) - dataset_mean.pow(2))

    print(f"Dataset Mean: {dataset_mean}")
    print(f"Dataset Stddev: {dataset_stddev}")

    print("Done")


if __name__ == "__main__":
    main()
