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


def generate_vae_output(dataset_root_dir: str, vae: BaseVAE, img_size: int, vae_output_file_name: str,
                        device: torch.device):
    dataset = PreprocessVAEDataset(dataset_root_dir, vae_transformation_functions(img_size))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    vae.eval()

    calculated_mus = []
    calculated_log_vars = []

    for data in tqdm(dataloader):
        data = data.to(device)
        with torch.no_grad():
            mu, log_var = vae(data)[1:]
            calculated_mus.append(mu)
            calculated_log_vars.append(log_var)

    calculated_mus = torch.cat(calculated_mus, dim=0)
    calculated_log_vars = torch.cat(calculated_log_vars, dim=0)

    with h5py.File(os.path.join(dataset_root_dir, vae_output_file_name), "w") as f:
        f.create_dataset("mus", data=calculated_mus)
        f.create_dataset("log_vars", data=calculated_log_vars)


def preprocess_observations_with_vae(dataset_path: str, vae: BaseVAE, vae_name: str, vae_version: int, img_size: int,
                                     device: torch.device, force: bool = False):
    dataset_path_content = os.listdir(dataset_path)
    dataset_path_content.sort()

    vae_output_file_name = f"preprocessed_data_vae_{vae_name}_version_{vae_version}.hdf5"

    if "observations" in dataset_path_content:
        # Single sequence folder given
        if vae_output_file_name not in dataset_path_content or force:
            generate_vae_output(dataset_path, vae, img_size, vae_output_file_name, device)
    else:
        for sequence in dataset_path_content:
            current_folder = os.path.join(dataset_path, sequence)
            if vae_output_file_name not in os.listdir(current_folder) or force:
                generate_vae_output(current_folder, vae, img_size, vae_output_file_name, device)
