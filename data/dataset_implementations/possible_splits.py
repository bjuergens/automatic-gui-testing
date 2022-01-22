from typing import Tuple

POSSIBLE_SPLITS = ["train", "val", "test"]

# Key value is the amount of sequences that were used in the data generation (located in root_dir), and the values is
# a list, which tell the end indices of the sequences that are used for the corresponding split. Therefore, the list has
# always a length of three and because we have [train, val, test] splits. See the first example down below for more
# detail.
IMPLEMENTED_SPLITS = {
    16: [10, 12, 16],  # e.g. 16 sequences, 0-10 for training, 10-12 for validation, 12-16 for testing
    10: [7, 8, 10],  # sequences# : 7 training, 1 validation, 2 testing
    100: [65, 80, 100]
}


def get_start_and_end_indices_from_split(number_of_sequences: int, split: str) -> Tuple[int, int]:
    assert number_of_sequences in IMPLEMENTED_SPLITS.keys()

    if split == "train":
        start_index = 0
        end_index = IMPLEMENTED_SPLITS[number_of_sequences][0]
    elif split == "val":
        start_index = IMPLEMENTED_SPLITS[number_of_sequences][0]
        end_index = IMPLEMENTED_SPLITS[number_of_sequences][1]
    elif split == "test":
        start_index = IMPLEMENTED_SPLITS[number_of_sequences][1]
        end_index = IMPLEMENTED_SPLITS[number_of_sequences][2]
    else:
        raise RuntimeError(f"Selected split {split} is not supported")

    return start_index, end_index
