from typing import Tuple

import torch
from torch import nn

from models.vae import BaseVAE


class DecreasingFilterSizesVAE(BaseVAE):

    def __init__(self, model_parameters: dict):
        super().__init__(model_parameters)

        assert len(self.hidden_dimensions) == 6, "For the chosen VAE architecture, 6 hidden dimensions are needed"

        conv_1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_dimensions[0], kernel_size=11, stride=2, padding=1)
        conv_2 = nn.Conv2d(in_channels=self.hidden_dimensions[0], out_channels=self.hidden_dimensions[1], kernel_size=9, stride=2, padding=0)
        conv_3 = nn.Conv2d(in_channels=self.hidden_dimensions[1], out_channels=self.hidden_dimensions[2], kernel_size=7, stride=2, padding=0)
        conv_4 = nn.Conv2d(in_channels=self.hidden_dimensions[2], out_channels=self.hidden_dimensions[3], kernel_size=5, stride=2, padding=1)
        conv_5 = nn.Conv2d(in_channels=self.hidden_dimensions[3], out_channels=self.hidden_dimensions[4], kernel_size=3, stride=2, padding=1)
        conv_6 = nn.Conv2d(in_channels=self.hidden_dimensions[4], out_channels=self.hidden_dimensions[5], kernel_size=3, stride=2, padding=1)

        if self.use_batch_norm:
            self.conv_layer_1 = nn.Sequential(conv_1, nn.BatchNorm2d(self.hidden_dimensions[0]), self.activation_function())
            self.conv_layer_2 = nn.Sequential(conv_2, nn.BatchNorm2d(self.hidden_dimensions[1]), self.activation_function())
            self.conv_layer_3 = nn.Sequential(conv_3, nn.BatchNorm2d(self.hidden_dimensions[2]), self.activation_function())
            self.conv_layer_4 = nn.Sequential(conv_4, nn.BatchNorm2d(self.hidden_dimensions[3]), self.activation_function())
            self.conv_layer_5 = nn.Sequential(conv_5, nn.BatchNorm2d(self.hidden_dimensions[4]), self.activation_function())
            self.conv_layer_6 = nn.Sequential(conv_6, nn.BatchNorm2d(self.hidden_dimensions[5]), self.activation_function())
        else:
            self.conv_layer_1 = nn.Sequential(conv_1, self.activation_function())
            self.conv_layer_2 = nn.Sequential(conv_2, self.activation_function())
            self.conv_layer_3 = nn.Sequential(conv_3, self.activation_function())
            self.conv_layer_4 = nn.Sequential(conv_4, self.activation_function())
            self.conv_layer_5 = nn.Sequential(conv_5, self.activation_function())
            self.conv_layer_6 = nn.Sequential(conv_6, self.activation_function())

        # Bottleneck Size: 6x6
        self.fc_mu = nn.Linear(6 * 6 * self.hidden_dimensions[5], self.latent_size)
        self.fc_log_var = nn.Linear(6 * 6 * self.hidden_dimensions[5], self.latent_size)

        self.fc_decoder = nn.Linear(self.latent_size, 6 * 6 * self.hidden_dimensions[5])

        transposed_conv_1 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[5], out_channels=self.hidden_dimensions[4], kernel_size=3, stride=2, padding=1, output_padding=1)
        transposed_conv_2 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[4], out_channels=self.hidden_dimensions[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        transposed_conv_3 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[3], out_channels=self.hidden_dimensions[2], kernel_size=5, stride=2, padding=1, output_padding=1)
        transposed_conv_4 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[2], out_channels=self.hidden_dimensions[1], kernel_size=7, stride=2, padding=0, output_padding=1)
        transposed_conv_5 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[1], out_channels=self.hidden_dimensions[0], kernel_size=9, stride=2, padding=0, output_padding=1)
        transposed_conv_6 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[0], out_channels=self.input_channels, kernel_size=11, stride=2, padding=1, output_padding=1)

        if self.use_batch_norm:
            self.transposed_conv_layer_1 = nn.Sequential(transposed_conv_1, nn.BatchNorm2d(self.hidden_dimensions[4]), self.activation_function())
            self.transposed_conv_layer_2 = nn.Sequential(transposed_conv_2, nn.BatchNorm2d(self.hidden_dimensions[3]), self.activation_function())
            self.transposed_conv_layer_3 = nn.Sequential(transposed_conv_3, nn.BatchNorm2d(self.hidden_dimensions[2]), self.activation_function())
            self.transposed_conv_layer_4 = nn.Sequential(transposed_conv_4, nn.BatchNorm2d(self.hidden_dimensions[1]), self.activation_function())
            self.transposed_conv_layer_5 = nn.Sequential(transposed_conv_5, nn.BatchNorm2d(self.hidden_dimensions[0]), self.activation_function())
            self.transposed_conv_layer_6 = nn.Sequential(transposed_conv_6, nn.BatchNorm2d(self.input_channels), self.output_activation_function())
        else:
            self.transposed_conv_layer_1 = nn.Sequential(transposed_conv_1, self.activation_function())
            self.transposed_conv_layer_2 = nn.Sequential(transposed_conv_2, self.activation_function())
            self.transposed_conv_layer_3 = nn.Sequential(transposed_conv_3, self.activation_function())
            self.transposed_conv_layer_4 = nn.Sequential(transposed_conv_4, self.activation_function())
            self.transposed_conv_layer_5 = nn.Sequential(transposed_conv_5, self.activation_function())
            self.transposed_conv_layer_6 = nn.Sequential(transposed_conv_6, self.output_activation_function())

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        x = self.conv_layer_6(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decoder(z)
        x = x.view(-1, self.hidden_dimensions[5], 6, 6)
        x = self.transposed_conv_layer_1(x)
        x = self.transposed_conv_layer_2(x)
        x = self.transposed_conv_layer_3(x)
        x = self.transposed_conv_layer_4(x)
        x = self.transposed_conv_layer_5(x)
        x = self.transposed_conv_layer_6(x)

        return x


def main():
    from torchinfo import summary

    model = DecreasingFilterSizesVAE({
        "input_channels": 3,
        "latent_size": 32,
        "hidden_dimensions": [8, 16, 32, 64, 128, 256],
        "activation_function": "leaky_relu",
        "batch_norm": True,
        "kld_warmup": True,
        "kld_weight": 1.0
    })
    summary(model, input_size=(2, 3, 448, 448))


if __name__ == "__main__":
    main()
