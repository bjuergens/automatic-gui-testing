import collections
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data.dataset_implementations import GUIMultipleSequencesObservationDataset, GUISequenceBatchSampler


"""
These are old tests, that were used during development for performance tests, etc.

Here for future reference, but they most certainly do not work anymore

"""


def custom_collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate_fn(samples) for samples in transposed]


def gpu_performance_test_sub_sequences():
    device = torch.device("cpu")

    transformation_functions = transforms.Compose([
        transforms.ToTensor()
    ])

    additional_dataloader_args = {"num_workers": 0, "pin_memory": True}

    dataset = GUIMultipleSequencesDataset(
        "datasets/gui_env/random-widgets/2021-12-31_12-47-33",
        # "datasets/gui_env/random-widgets/2021-12-29_19-02-29",
        train=True,
        sequence_length=16,
        transform=transformation_functions
    )

    custom_sampler = GUISequenceBatchSampler(dataset, batch_size=32, drop_last=True)

    dataloader = DataLoader(
        dataset,
        batch_sampler=custom_sampler,
        # collate_fn=custom_collate_fn,
        **additional_dataloader_args
    )

    start_time = time.time()
    for current_epoch in range(2):
        for batch_id, data in enumerate(dataloader):

            batch_data = data

            if batch_id % 20 == 0:
                print(f"Epoch {current_epoch} - Batch ID {batch_id}")

            if batch_id > 100:
                break

    print(f"Took time: {time.time() - start_time}")  # 12sec 6 worker 17 sec 2 worker 13sec 8 worker 12sec 0 worker


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


def observation_sequence_dataset_init_test():

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


def main():

    gpu_performance_test_sub_sequences()

    print("Done")


if __name__ == "__main__":
    main()
