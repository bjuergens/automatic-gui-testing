import torch
import torch.nn as nn

from models.rnn import BaseSimpleRNN


class SimpleLSTM(BaseSimpleRNN):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        self.rnn = nn.LSTM(input_size=self.latent_size + self.action_size, hidden_size=self.hidden_size,
                           batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.latent_size + 1)
