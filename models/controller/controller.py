import torch
import torch.nn as nn

from utils.constants import MAX_COORDINATE


class Controller(nn.Module):

    def __init__(self, latent_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.fc = nn.Linear(latent_size + hidden_size, action_size)

        # Output of Controller is in tanh range ([-1, 1]) and we want to predict actions in the range of
        # [0, MAX_COORDINATE] as the environment has this action space (MAX_COORDINATE is 448)
        self.map_actions_to_action_range = lambda x: ((x + 1.0) * (MAX_COORDINATE - 1.0)) / 2.0

    def forward(self, latent_observation: torch.Tensor, hidden_state: torch.Tensor):
        x = torch.cat([latent_observation.squeeze(), hidden_state.squeeze()], dim=0)
        x = self.fc(x)
        x = torch.tanh(x)
        return x

    def predict(self, model_output):
        x = self.map_actions_to_action_range(model_output).round().int().view(1, 1, -1)
        return x
