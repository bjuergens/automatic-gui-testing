import os

import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.vae import BaseVAE
from utils.training_utils import vae_transformation_functions


class PreprocessVAEDataset(Dataset):

    def __init__(self, root_dir, transform_functions):
        self.root_dir = root_dir
        self.transform_functions = transform_functions

        self.image_files = os.listdir(os.path.join(self.root_dir, "observations"))
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, "observations", self.image_files[index]))
        img = self.transform_functions(img)
        return img


def get_vae_preprocessed_data_path_name(vae_directory: str, rnn_dataset_name: str):
    return os.path.join(vae_directory, f"vae_preprocessed_data_for_{rnn_dataset_name}.hdf5")


def generate_vae_output(dataset_root_dir: str, vae: BaseVAE, img_size: int, vae_dataset_name: str,
                        output_activation_function: str, vae_preprocessed_data_path: str, hdf5_data_group_path: str,
                        device: torch.device):

    transformation_functions = vae_transformation_functions(img_size, vae_dataset_name, output_activation_function)
    dataset = PreprocessVAEDataset(dataset_root_dir, transformation_functions)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=6, pin_memory=True, drop_last=False)
    vae.eval()

    calculated_mus = []
    calculated_log_vars = []

    for data in tqdm(dataloader):
        data = data.to(device)
        with torch.no_grad():
            mu, log_var = vae(data)[1:]
            calculated_mus.append(mu)
            calculated_log_vars.append(log_var)

    calculated_mus = torch.cat(calculated_mus, dim=0).cpu()
    calculated_log_vars = torch.cat(calculated_log_vars, dim=0).cpu()

    with h5py.File(vae_preprocessed_data_path, "a") as f:
        group = f.create_group(hdf5_data_group_path)
        group.create_dataset(f"{hdf5_data_group_path}/mus", data=calculated_mus)
        group.create_dataset(f"{hdf5_data_group_path}/log_vars", data=calculated_log_vars)


def preprocess_observations_with_vae(rnn_dataset_path: str, vae: BaseVAE, img_size: int,
                                     output_activation_function: str, vae_dataset_name, device: torch.device,
                                     vae_preprocessed_data_path: str):
    dataset_path_content = os.listdir(rnn_dataset_path)
    dataset_path_content.sort()

    # Folder structure: root_dir/split/sequence_lengths/sequences
    for sub_dir_split in dataset_path_content:
        sub_dir_split_path = os.path.join(rnn_dataset_path, sub_dir_split)
        sub_dir_split_content = os.listdir(sub_dir_split_path)
        sub_dir_split_content.sort()

        for sub_dir_sequence_length in sub_dir_split_content:
            sub_dir_sequence_length_path = os.path.join(sub_dir_split_path, sub_dir_sequence_length)
            sub_dir_sequence_length_content = os.listdir(sub_dir_sequence_length_path)
            sub_dir_sequence_length_content.sort()

            for sequence_dir in sub_dir_sequence_length_content:
                sequence_dir_path = os.path.join(sub_dir_sequence_length_path, sequence_dir)
                hdf5_data_group_path = f"/{sub_dir_split}/{sub_dir_sequence_length}/{sequence_dir}"

                generate_vae_output(
                    sequence_dir_path,
                    vae,
                    img_size,
                    vae_dataset_name,
                    output_activation_function,
                    vae_preprocessed_data_path,
                    hdf5_data_group_path,
                    device
                )
