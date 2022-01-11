from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data.gui_dataset import GUIMultipleSequencesObservationDataset, GUISequenceBatchSampler, \
    GUIMultipleSequencesDataset


def iteration_test():
    batch_size = 2
    sequence_length = 3

    transformation_functions = transforms.Compose([
        transforms.Resize((8, 8)),  # Saves time in this test
        transforms.ToTensor()
    ])

    dataset_path = "datasets/gui_env/random-widgets/2021-12-29_19-02-29"
    dataset = GUIMultipleSequencesDataset(dataset_path, train=True, sequence_length=sequence_length,
                                          transform=transformation_functions)

    custom_sampler = GUISequenceBatchSampler(dataset, batch_size=batch_size, drop_last=True)

    dataloader = DataLoader(
        dataset,
        batch_sampler=custom_sampler
    )

    max_epochs = 2

    for current_epoch in range(max_epochs):
        print(f"Started Epoch {current_epoch}")
        for batch_id, data in enumerate(dataloader):
            batch_data, dataset_index = data

            assertion = all([i == dataset_index[0] for i in dataset_index])

            if batch_id % 20 == 0:
                print(f"Epoch {current_epoch} - Batch ID {batch_id}")


def main():

    transformation_functions = transforms.Compose([
        transforms.Resize((8, 8)),  # Saves time in this test
        transforms.ToTensor()
    ])

    train_dataset = GUIMultipleSequencesObservationDataset(
        root_dir="datasets/gui_env/random-widgets/2021-12-31_12-47-33",
        split="train",
        transform=transformation_functions
    )

    val_dataset = GUIMultipleSequencesObservationDataset(
        root_dir="datasets/gui_env/random-widgets/2021-12-31_12-47-33",
        split="val",
        transform=transformation_functions
    )

    test_dataset = GUIMultipleSequencesObservationDataset(
        root_dir="datasets/gui_env/random-widgets/2021-12-31_12-47-33",
        split="test",
        transform=transformation_functions
    )

    assert len(train_dataset) == 513*7
    assert len(val_dataset) == 513*1
    assert len(test_dataset) == 513*2

    iteration_test()

    print("Done")


if __name__ == "__main__":
    main()
