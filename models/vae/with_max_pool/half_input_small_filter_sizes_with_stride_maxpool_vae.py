from typing import Tuple

import torch
from torch import nn

from models.vae import BaseVAE


class HalfInputSmallFilterSizesWithStrideMaxPoolVAE(BaseVAE):

    def __init__(self, model_parameters: dict):
        super().__init__(model_parameters)

        assert len(self.hidden_dimensions) == 4, "For the chosen VAE architecture, 6 hidden dimensions are needed"

        conv_1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_dimensions[0], kernel_size=3, stride=2, padding=1)
        conv_2 = nn.Conv2d(in_channels=self.hidden_dimensions[0], out_channels=self.hidden_dimensions[1], kernel_size=3, stride=2, padding=1)
        conv_3 = nn.Conv2d(in_channels=self.hidden_dimensions[1], out_channels=self.hidden_dimensions[2], kernel_size=3, stride=2, padding=1)
        conv_4 = nn.Conv2d(in_channels=self.hidden_dimensions[2], out_channels=self.hidden_dimensions[3], kernel_size=3, stride=1, padding=0)

        if self.use_batch_norm:
            self.conv_layer_1 = nn.Sequential(conv_1, nn.BatchNorm2d(self.hidden_dimensions[0]), self.activation_function(), nn.MaxPool2d(2, 2))
            self.conv_layer_2 = nn.Sequential(conv_2, nn.BatchNorm2d(self.hidden_dimensions[1]), self.activation_function(), nn.MaxPool2d(2, 2))
            self.conv_layer_3 = nn.Sequential(conv_3, nn.BatchNorm2d(self.hidden_dimensions[2]), self.activation_function(), nn.MaxPool2d(2, 2))
            self.conv_layer_4 = nn.Sequential(conv_4, nn.BatchNorm2d(self.hidden_dimensions[3]), self.activation_function())
        else:
            self.conv_layer_1 = nn.Sequential(conv_1, self.activation_function(), nn.MaxPool2d(2, 2))
            self.conv_layer_2 = nn.Sequential(conv_2, self.activation_function(), nn.MaxPool2d(2, 2))
            self.conv_layer_3 = nn.Sequential(conv_3, self.activation_function(), nn.MaxPool2d(2, 2))
            self.conv_layer_4 = nn.Sequential(conv_4, self.activation_function())

        # Bottleneck 6x6 when using MaxPool(2, 2) and at the last MaxPool a filter size of 3
        self.fc_mu = nn.Linear(1 * 1 * self.hidden_dimensions[3], self.latent_size)
        self.fc_log_var = nn.Linear(1 * 1 * self.hidden_dimensions[3], self.latent_size)

        self.fc_decoder = nn.Linear(self.latent_size, 1 * 1 * self.hidden_dimensions[3])

        transposed_conv_1 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[3], out_channels=self.hidden_dimensions[2], kernel_size=3, stride=2, padding=0, output_padding=0)
        transposed_conv_2 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[2], out_channels=self.hidden_dimensions[1], kernel_size=5, stride=4, padding=0, output_padding=1)
        transposed_conv_3 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[1], out_channels=self.hidden_dimensions[0], kernel_size=5, stride=4, padding=1, output_padding=1)
        transposed_conv_4 = nn.ConvTranspose2d(in_channels=self.hidden_dimensions[0], out_channels=self.input_channels, kernel_size=5, stride=4, padding=1, output_padding=1)

        if self.use_batch_norm:
            self.transposed_conv_layer_1 = nn.Sequential(transposed_conv_1, nn.BatchNorm2d(self.hidden_dimensions[2]), self.activation_function())
            self.transposed_conv_layer_2 = nn.Sequential(transposed_conv_2, nn.BatchNorm2d(self.hidden_dimensions[1]), self.activation_function())
            self.transposed_conv_layer_3 = nn.Sequential(transposed_conv_3, nn.BatchNorm2d(self.hidden_dimensions[0]), self.activation_function())
            self.transposed_conv_layer_4 = nn.Sequential(transposed_conv_4, nn.BatchNorm2d(self.input_channels), self.output_activation_function())
        else:
            self.transposed_conv_layer_1 = nn.Sequential(transposed_conv_1, self.activation_function())
            self.transposed_conv_layer_2 = nn.Sequential(transposed_conv_2, self.activation_function())
            self.transposed_conv_layer_3 = nn.Sequential(transposed_conv_3, self.activation_function())
            self.transposed_conv_layer_4 = nn.Sequential(transposed_conv_4, self.output_activation_function())

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decoder(z)
        x = x.view(-1, self.hidden_dimensions[3], 1, 1)
        x = self.transposed_conv_layer_1(x)
        x = self.transposed_conv_layer_2(x)
        x = self.transposed_conv_layer_3(x)
        x = self.transposed_conv_layer_4(x)

        return x


def main():
    from torchinfo import summary

    model = HalfInputSmallFilterSizesWithStrideMaxPoolVAE({
        "input_channels": 3,
        "latent_size": 32,
        "hidden_dimensions": [8, 16, 32, 64],
        "activation_function": "leaky_relu",
        "output_activation_function": "tanh",
        "batch_norm": True,
        "kld_warmup": True,
        "kld_weight": 1.0,
        "kld_warmup_batch_count": 20,
        "kld_warmup_skip_batches": 20
    })
    summary(model, input_size=(1, 3, 224, 224))


if __name__ == "__main__":
    main()
