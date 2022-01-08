from torchvision.transforms import transforms

from data.gui_dataset import GUIMultipleSequencesObservationDataset


def main():

    transformation_functions = transforms.Compose([
        transforms.Resize((224, 224)),
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

    print("Done")


if __name__ == "__main__":
    main()
