import torch
import torch.nn as nn

from models.rnn import BaseMDNRNN


class StandardMDNRNN(BaseMDNRNN):
    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        self.rnn = nn.LSTM(self.latent_size + self.action_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, (2 * self.latent_size + 1) * self.number_of_gaussians + 1)
