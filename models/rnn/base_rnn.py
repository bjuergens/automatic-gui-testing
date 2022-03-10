import abc
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f


ONE_OVER_SQRT_2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)

LOSS_SCALE_OPTION_MUL = "multiplication"
LOSS_SCALE_OPTION_DIV = "division"


class BaseRNN(abc.ABC, nn.Module):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__()

        self.hidden_size = model_parameters["hidden_size"]
        self.number_of_hidden_layers = model_parameters["hidden_layers"]
        self.action_size = model_parameters["action_size"]
        self.latent_size = latent_size
        self.loss_scale_option = model_parameters["loss_scale_option"]  # Can also be none

        self.reward_output_activation_function_type = model_parameters["reward_output_activation_function"]

        if self.reward_output_activation_function_type == "sigmoid":
            self.reward_output_activation_function = nn.Sigmoid()
            self.denormalize_reward = lambda x: x
        elif self.reward_output_activation_function_type == "tanh":
            self.reward_output_activation_function = nn.Tanh()
            self.denormalize_reward = lambda x: (x + 1.0) / 2.0
        else:
            raise RuntimeError(f"Output activation function {self.reward_output_activation_function_type} unknown")

        self.batch_size = batch_size
        self.device = device

        self.rnn = None
        self.fc = None
        self.hidden_state, self.cell_state = None, None

        self.initialize_hidden()

    def initialize_hidden(self):
        self.hidden_state = torch.zeros((self.number_of_hidden_layers, self.batch_size, self.hidden_size),
                                        device=self.device, requires_grad=True)
        self.cell_state = torch.zeros((self.number_of_hidden_layers, self.batch_size, self.hidden_size),
                                      device=self.device, requires_grad=True)

        return (self.hidden_state, self.cell_state)

    def rnn_forward(
            self, latents: torch.Tensor, actions: torch.Tensor,
            hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.cat([latents, actions], dim=-1)
        outputs, new_hidden = self.rnn(x, hidden)

        return outputs, new_hidden

    def combine_latent_and_reward_loss(self, latent_loss, reward_loss):
        if self.loss_scale_option is None:
            loss = latent_loss + reward_loss
        elif self.loss_scale_option == LOSS_SCALE_OPTION_MUL:
            # Scale is calculated as latent size + 1 (for the reward). This was done in the implementation this
            # repository is based on
            scale = self.latent_size
            loss = latent_loss + (reward_loss * scale)
        elif self.loss_scale_option == LOSS_SCALE_OPTION_DIV:
            scale = self.latent_size + 1
            loss = (latent_loss + reward_loss) / scale
        else:
            raise RuntimeError(f"Loss scale option '{self.loss_scale_option}' unknown")

        return loss

    @abc.abstractmethod
    def predict(self, model_output, latents) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(self, latent_vector: torch.Tensor, action: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        pass

    @abc.abstractmethod
    def loss_function(self, next_latent_vector: torch.Tensor, reward: torch.Tensor, model_output: Tuple):
        pass

    @abc.abstractmethod
    def get_reward_output_mode(self) -> str:
        pass


class BaseMDNRNN(BaseRNN):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        self.number_of_gaussians = model_parameters["number_of_gaussians"]

    def predict(self, model_output, latents):
        mus = model_output[0]
        sigmas = model_output[1]
        log_pi = model_output[2]
        rewards = model_output[3]

        # batch: (BATCH_SIZE, SEQ_LEN, 1, L_SIZE)
        # mus, sigmas: (BATCH_SIZE, SEQ_LEN, N_GAUSS, L_SIZE)
        # log_pi: (BATCH_SIZE, SEQ_LEN, N_GAUSS)
        latents = latents.unsqueeze(-2)
        # prob: (BATCH_SIZE, SEQ_LEN, N_GAUSS, L_SIZE)
        prob = ONE_OVER_SQRT_2PI * torch.exp(-0.5 * torch.pow((latents - mus) / sigmas, 2)) / sigmas
        # log_pi: (BATCH_SIZE, SEQ_LEN, N_GAUSS, 1)
        pi = log_pi.exp().unsqueeze(-1)

        latent_prediction = torch.sum(prob * pi, dim=2)

        # Result: (BATCH_SIZE, SEQ_LEN, L_SIZE)
        return latent_prediction, self.denormalize_reward(rewards)

    def forward(self,
                latents: torch.Tensor,
                actions: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor]):
        sequence_length = latents.size(1)

        outputs, new_hidden = self.rnn_forward(latents, actions, hidden)
        gmm_outputs = self.fc(outputs)

        stride = self.number_of_gaussians * self.latent_size

        mus = gmm_outputs[:, :, :stride]
        mus = mus.view(self.batch_size, sequence_length, self.number_of_gaussians, self.latent_size)

        sigmas = gmm_outputs[:, :, stride:2 * stride]
        sigmas = sigmas.view(self.batch_size, sequence_length, self.number_of_gaussians, self.latent_size)
        sigmas = torch.exp(sigmas)

        pi = gmm_outputs[:, :, 2 * stride: 2 * stride + self.number_of_gaussians]
        pi = pi.view(self.batch_size, sequence_length, self.number_of_gaussians)
        log_pi = f.log_softmax(pi, dim=-1)

        rewards = gmm_outputs[:, :, -1]
        rewards = self.reward_output_activation_function(rewards).unsqueeze(-1)

        return (mus, sigmas, log_pi, rewards), new_hidden

    def _gmm_loss_using_log(self, batch, mus, sigmas, log_pi, reduce=True):
        log_prob = -torch.log(sigmas) - 0.5 * LOG2PI - 0.5 * torch.pow((batch - mus) / sigmas, 2)
        log_prob_sum = log_prob.sum(dim=-1)
        log_prob_sum = torch.logsumexp(log_pi + log_prob_sum, dim=-1)

        if reduce:
            nll = -log_prob_sum.mean()
        else:
            nll = -log_prob_sum

        return nll

    def gmm_loss(self, next_latent_vector, mus, sigmas, log_pi):
        next_latent_vector = next_latent_vector.unsqueeze(-2)
        prob = ONE_OVER_SQRT_2PI * torch.exp(-0.5 * torch.pow((next_latent_vector - mus) / sigmas, 2)) / sigmas

        eps = 1e-10
        log_prob = torch.log(prob + eps).sum(dim=-1)
        log_prob = torch.logsumexp(log_pi + log_prob, dim=-1)

        nll = -log_prob.mean()

        return nll

    def loss_function(self, next_latent_vector: torch.Tensor, reward: torch.Tensor, model_output: Tuple):
        mus = model_output[0]
        sigmas = model_output[1]
        log_pi = model_output[2]
        predicted_reward = model_output[3]

        gmm = self.gmm_loss(next_latent_vector, mus, sigmas, log_pi)
        mse = f.mse_loss(predicted_reward, reward)

        loss = self.combine_latent_and_reward_loss(latent_loss=gmm, reward_loss=mse)

        return loss, (gmm.item(), mse.item())

    def get_reward_output_mode(self) -> str:
        return "mse"


class BaseSimpleRNN(BaseRNN):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

    def predict(self, model_output, latents=None):
        # This function mostly exists for mixture density network as the actual calculation of the next latent state
        # is not required for training, just the calculation of the predicted probability distribution.
        # But since we want to use the same interface, just return the prediction here
        return model_output[0], self.denormalize_reward(model_output[1])

    def forward(self, latents: torch.Tensor, actions: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        outputs, new_hidden = self.rnn_forward(latents, actions, hidden)

        predictions = self.fc(outputs)

        predicted_latent_vector = predictions[:, :, :self.latent_size]
        predicted_reward = self.reward_output_activation_function(predictions[:, :, self.latent_size:])

        return (predicted_latent_vector, predicted_reward), new_hidden

    def loss_function(self, next_latent_vector: torch.Tensor, reward: torch.Tensor, model_output: Tuple):
        predicted_latent, predicted_reward = model_output[0], model_output[1]

        # TODO check if reduction needs to be adapted to batch size and sequence length
        latent_loss = f.mse_loss(predicted_latent, next_latent_vector)
        reward_loss = f.mse_loss(predicted_reward, reward)

        loss = self.combine_latent_and_reward_loss(latent_loss=latent_loss, reward_loss=reward_loss)

        return loss, (latent_loss.item(), reward_loss.item())

    def get_reward_output_mode(self) -> str:
        return "mse"
