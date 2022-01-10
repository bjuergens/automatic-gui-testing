import abc
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as f


class BaseVAE(abc.ABC, nn.Module):

    def __init__(self, model_parameters: dict, use_kld_warmup: bool, kld_weight: float = 1.0):
        super().__init__()

        self.input_channels = model_parameters["input_channels"]
        self.latent_size = model_parameters["latent_size"]

        self.use_kld_warmup = use_kld_warmup
        self.kld_weight = kld_weight

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        return z

    def sample(self, number_of_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn((number_of_samples, self.latent_size)).to(device)
        return self.decode(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)

        return reconstruction, mu, log_var

    def loss_function(self, x: torch.Tensor, reconstruction_x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
                      current_epoch: int, max_epochs: int) -> Tuple[torch.Tensor, float, float]:
        # MSE
        batch_dim = x.size(0)
        reconstruction_loss = f.mse_loss(x, reconstruction_x, reduction="sum") / batch_dim

        # KLD
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Take also the mean over the batch_dim (outermost function call)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

        if self.use_kld_warmup:
            kld_warmup_term = current_epoch / max_epochs
            kld_loss_term = self.kld_weight * kld_warmup_term * kld_loss
        else:
            kld_loss_term = self.kld_weight * kld_loss

        loss = reconstruction_loss + kld_loss_term

        # .item() is important as it extracts a float, otherwise the tensors would be held in memory and never freed
        reconstruction_loss_float = reconstruction_loss.item()
        kld_loss_float = kld_loss.item()

        return loss, reconstruction_loss_float, kld_loss_float
