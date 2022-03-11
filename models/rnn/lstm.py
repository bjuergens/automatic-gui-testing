from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.rnn import BaseSimpleRNN


class LSTM(BaseSimpleRNN):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        self.rnn = nn.LSTM(input_size=self.latent_size + self.action_size, hidden_size=self.hidden_size,
                           batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.latent_size + 1)


class LSTMWithBCE(LSTM):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        assert model_parameters["reward_output_activation_function"] == "sigmoid", ("For this RNN the output "
                                                                                    "activation function has to be the "
                                                                                    "sigmoid function")

        # Supports float and torch.Tensor objects. As we train with either 0 or 1 as a reward we also want to predict
        # that.
        self.denormalize_reward = lambda x: int(x > 0.5)

    def predict(self, model_output, latents=None):
        # Apply sigmoid here to reward instead of self.reward_output_activation_function, because we don't apply that
        # function in forward(), instead it is fused into the loss function for the reward. Still for the prediction
        # we want values in [0, 1] range for the reward
        # Also since we use BCE loss and train to predict either 0 or 1 reward we have to denormalize the reward
        # This function basically rounds down to 0 if <= 0.5 or round up to 1 in the other case
        return model_output[0], self.denormalize_reward(torch.sigmoid(model_output[1]))

    def forward(self, latents: torch.Tensor, actions: torch.Tensor):
        outputs, _ = self.rnn_forward(latents, actions)

        predictions = self.fc(outputs)

        predicted_latent_vector = predictions[:, :, :self.latent_size]
        # Don't use self.reward_output_activation_function here, because we use BCEWithLogitsLoss which applies a
        # sigmoid internally (see a bit below for more information)
        predicted_reward = predictions[:, :, self.latent_size:]

        return predicted_latent_vector, predicted_reward

    def loss_function(self, next_latent_vector: torch.Tensor, reward: torch.Tensor, model_output: Tuple):
        predicted_latent, predicted_reward = model_output[0], model_output[1]

        # TODO check if reduction needs to be adapted to batch size and sequence length
        latent_loss = f.mse_loss(predicted_latent, next_latent_vector)

        # Computes sigmoid followed by BCELoss. Is numerically more stable than doing the two steps separately, see
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        reward_loss = f.binary_cross_entropy_with_logits(predicted_reward, reward)
        loss = self.combine_latent_and_reward_loss(latent_loss=latent_loss, reward_loss=reward_loss)

        return loss, (latent_loss.item(), reward_loss.item())

    def get_reward_output_mode(self) -> str:
        return "bce"


class LSTMWithMSE(LSTM):

    def __init(self, **kwargs):
        super().__init__(**kwargs)
