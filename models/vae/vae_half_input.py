from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae.base_vae import BaseVAE


class VAEHalfInputSize(BaseVAE):

    def __init__(self, model_parameters: dict, use_kld_warmup: bool, kld_weight: float = 1.0):
        super().__init__(model_parameters, use_kld_warmup, kld_weight)

        # Encoder
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

        self.fc_mu = nn.Linear(2*2*256, self.latent_size)
        self.fc_log_var = nn.Linear(2*2*256, self.latent_size)

        # Decoder
        self.fc_decoder = nn.Linear(self.latent_size, 256 * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)  # 6
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2)  # 13
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)   # 27
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)  # 55
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)  # 111
        self.deconv6 = nn.ConvTranspose2d(32, self.img_channels, kernel_size=4, stride=2)  # 224

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_deccoder(z))
        x = x.view(-1, 256, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))

        reconstruction = torch.sigmoid(self.deconv6(x))

        return reconstruction
