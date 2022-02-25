import abc
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f


class BaseVAE(abc.ABC, nn.Module):

    def __init__(self, model_parameters: dict):
        super().__init__()

        self.input_channels = model_parameters["input_channels"]
        self.latent_size = model_parameters["latent_size"]
        self.hidden_dimensions = model_parameters["hidden_dimensions"]

        activation_function = model_parameters["activation_function"]

        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "leaky_relu":
            self.activation_function = nn.LeakyReLU
        else:
            raise RuntimeError(f"Activation function {activation_function} unknown")

        output_activation_function = model_parameters["output_activation_function"]

        if output_activation_function == "sigmoid":
            self.output_activation_function = nn.Sigmoid
            self.denormalize = lambda x: x
        elif output_activation_function == "tanh":
            self.output_activation_function = nn.Tanh
            self.denormalize = lambda x: (x + 1.0) / 2.0
        else:
            raise RuntimeError(f"Output activation function {output_activation_function} unknown")

        self.use_batch_norm = model_parameters["batch_norm"]

        try:
            self.disable_kld = model_parameters["disable_kld"]
        except KeyError:
            self.disable_kld = False

        try:
            self.apply_value_range_when_kld_disabled = model_parameters["apply_value_range_when_kld_disabled"]
        except KeyError:
            self.apply_value_range_when_kld_disabled = False

        self.use_kld_warmup = model_parameters["kld_warmup"]
        self.kld_weight = model_parameters["kld_weight"]
        self.kld_warmup_batch_count = model_parameters["kld_warmup_batch_count"]
        self.kld_warmup_skip_batches = model_parameters["kld_warmup_skip_batches"]
        self.current_batch_count = 0

        try:
            self.reduce_kld_weight_after_batch_count = model_parameters["reduce_kld_weight_after_batch_count"]
        except KeyError:
            self.reduce_kld_weight_after_batch_count = np.inf

        if self.use_kld_warmup is None and self.kld_weight is None:
            raise RuntimeError(f"kld_warmup and kld_weight parameters not in config, maybe you are using an older "
                               "model architecture.")

        if self.use_kld_warmup:
            assert self.kld_warmup_batch_count > 0, "When using KLD warm-up the kld_warmup_batch_count cannot be 0"

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def reparameterization_trick(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        return z

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        z = self.reparameterization_trick(mu, log_var)

        if self.disable_kld and self.apply_value_range_when_kld_disabled:
            z = torch.tanh(z)

        return z

    def sample(self, number_of_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn((number_of_samples, self.latent_size)).to(device)

        if self.disable_kld and self.apply_value_range_when_kld_disabled:
            z = torch.tanh(z)

        return self.decode(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)

        return reconstruction, mu, log_var

    def loss_function(self, x: torch.Tensor, reconstruction_x: torch.Tensor, mu: torch.Tensor,
                      log_var: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, float, float]:
        # MSE
        reconstruction_loss = f.mse_loss(x, reconstruction_x, reduction="mean")

        if not self.disable_kld:

            if self.current_batch_count > self.reduce_kld_weight_after_batch_count:
                self.kld_weight = 0.00001

            # KLD
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # Take also the mean over the batch_dim (outermost function call)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

            if self.use_kld_warmup and train:
                if self.current_batch_count < self.kld_warmup_skip_batches:
                    kld_warmup_term = 0.0
                else:
                    kld_warmup_term = self.current_batch_count / self.kld_warmup_batch_count

                    if kld_warmup_term > 1.0:
                        kld_warmup_term = 1.0

                kld_loss_term = self.kld_weight * kld_warmup_term * kld_loss

                self.current_batch_count += 1
            else:
                kld_loss_term = self.kld_weight * kld_loss

            loss = reconstruction_loss + kld_loss_term
            kld_loss_float = kld_loss.item()
        else:
            loss = reconstruction_loss
            kld_loss_float = 0.0

        # .item() is important as it extracts a float, otherwise the tensors would be held in memory and never freed
        # Log actual KLD loss and not the kld_loss_term which is what is used to calculate the whole loss function
        return loss, reconstruction_loss.item(), kld_loss_float
