from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.rnn import BaseMDNRNN


class StandardMDNRNN(BaseMDNRNN):
    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        self.rnn = nn.LSTM(self.latent_size + self.action_size, self.hidden_size, batch_first=True)

        if self.use_gaussian_per_latent_dim:
            # Each dimension of the latent space will be modelled by number_of_gaussian gaussians
            self.fc = nn.Linear(self.hidden_size, 3 * self.latent_size * self.number_of_gaussians + 1)
        else:
            self.fc = nn.Linear(self.hidden_size, (2 * self.latent_size + 1) * self.number_of_gaussians + 1)


class MDNRNNWithBCE(StandardMDNRNN):
    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        assert model_parameters["reward_output_activation_function"] == "sigmoid", ("For this RNN the output "
                                                                                    "activation function has to be the "
                                                                                    "sigmoid function")

        # Supports float and torch.Tensor objects. As we train with either 0 or 1 as a reward we also want to predict
        # that.
        self.denormalize_reward = lambda x: torch.round(x).int()

    def predict(self, model_output, latents, temperature):
        latent_prediction = self._predict_gaussian_mixture(model_output, temperature)
        predicted_reward_in_logits = model_output[3]

        # latent_prediction: (BATCH_SIZE, SEQ_LEN, L_SIZE)
        return latent_prediction, self.denormalize_reward(torch.sigmoid(predicted_reward_in_logits))

    def forward(self, latents: torch.Tensor, actions: torch.Tensor):
        sequence_length = latents.size(1)

        outputs, _ = self.rnn_forward(latents, actions)
        gmm_outputs = self.fc(outputs)

        # Difference to StandardMDN: No activation function on reward, as the sigmoid applied on it is included in the
        # BCELoss in the Loss Function for numerical stability
        mus, sigmas, log_pi, rewards = self._forward_gaussian_mixture(gmm_outputs, sequence_length)

        return mus, sigmas, log_pi, rewards

    def loss_function(self, next_latent_vector: torch.Tensor, reward: torch.Tensor, model_output: Tuple):
        mus = model_output[0]
        sigmas = model_output[1]
        log_pi = model_output[2]
        predicted_reward_in_logits = model_output[3]

        gmm = self.gmm_loss(next_latent_vector, mus, sigmas, log_pi)
        reward_loss = f.binary_cross_entropy_with_logits(predicted_reward_in_logits, reward)

        loss = self.combine_latent_and_reward_loss(latent_loss=gmm, reward_loss=reward_loss)

        return loss, (gmm.item(), reward_loss.item())

    @staticmethod
    def get_reward_output_mode() -> str:
        return "bce"
