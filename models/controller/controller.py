import torch
import torch.nn as nn

from utils.constants import MAX_COORDINATE


class Controller(nn.Module):

    def __init__(self, latent_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.fc = nn.Linear(latent_size + hidden_size, action_size)

    def forward(self, latent_observation: torch.Tensor, hidden_state: torch.Tensor):
        x = torch.cat([latent_observation.squeeze(), hidden_state.squeeze()], dim=0)
        x = self.fc(x)
        x = torch.tanh(x)
        return x

    def predict(self, model_output):
        """
        Map [-1, 1] range of model_output (coming from tanh in forward()) to integers in [0, MAX_COORDINATE - 1.0] range
        as this is what the controller should predict to navigate in the SUT (GUIEnv)
        """
        x = model_output.add(1.0)
        # -1 because we start counting by 0 (pixel coordinates)
        x = x.mul(MAX_COORDINATE - 1)
        x = torch.div(x, 2.0)
        x = x.round().int().view(1, 1, -1)
        return x
