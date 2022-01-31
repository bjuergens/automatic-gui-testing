import torch
import torch.nn as nn

from models.vae.base_vae import BaseVAE


class VAEFullInputSize(BaseVAE):

    def __init__(self, model_parameters: dict):
        super().__init__(model_parameters)

        assert self.hidden_dimensions == [32, 64, 128, 256], "For this VAE the hidden dimensions are fixed"
        assert self.activation_function == nn.LeakyReLU

        # Encoder
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(2*2*256, self.latent_size)
        self.fc_log_var = nn.Linear(2*2*256, self.latent_size)

        # Decoder
        self.fc_decoder = nn.Linear(self.latent_size, 2*2*256)

        self.transposed_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=4, padding=1),
            nn.LeakyReLU()
        )

        self.transposed_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

        self.transposed_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

        self.transposed_conv_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=self.input_channels, kernel_size=5, stride=4, padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

    def decode(self, z: torch.Tensor):
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 256, 2, 2)

        x = self.transposed_conv_1(x)
        x = self.transposed_conv_2(x)
        x = self.transposed_conv_3(x)
        x = self.transposed_conv_4(x)

        return x
