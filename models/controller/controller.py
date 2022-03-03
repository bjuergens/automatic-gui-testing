import torch
import torch.nn as nn


class Controller(nn.Module):

    def __init__(self, latent_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.fc = nn.Linear(latent_size + hidden_size, action_size)

    def forward(self, latent_observation: torch.Tensor, hidden_state: torch.Tensor):
        x = torch.cat([latent_observation.squeeze(), hidden_state.squeeze()], dim=0)
        x = self.fc(x)
        x = torch.tanh(x)
        x = x.unsqueeze(0).unsqueeze(0)
        return x
