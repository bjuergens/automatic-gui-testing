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

        # We generally assume that the data has generated clicks in the [0, 447] range for both x and y directions
        # With this parameter we allow to reduce this range by an amount, essentially grouping multiple pixels together
        # One click would then mean a click in all these pixels. This is of course false, but may help during training
        # to also cover previously not covered action spaces
        self.reduce_action_coordinate_space_by = model_parameters["reduce_action_coordinate_space_by"]
        self.action_transformation_function_type = model_parameters["action_transformation_function"]

        assert self.action_transformation_function_type in [None, "tanh"]

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
        self.hidden = None
        self.initialize_hidden()

    def initialize_hidden(self):
        hidden_state = torch.zeros((self.number_of_hidden_layers, self.batch_size, self.hidden_size),
                                   device=self.device, requires_grad=True)
        cell_state = torch.zeros((self.number_of_hidden_layers, self.batch_size, self.hidden_size),
                                 device=self.device, requires_grad=True)

        self.hidden = (hidden_state, cell_state)

        return self.hidden

    def rnn_forward(
            self, latents: torch.Tensor,
            actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latents, actions], dim=-1)
        outputs, self.hidden = self.rnn(x, self.hidden)

        return outputs, self.hidden

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
    def predict(self, model_output, latents, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(self, latent_vector: torch.Tensor, action: torch.Tensor):
        pass

    @abc.abstractmethod
    def loss_function(self, next_latent_vector: torch.Tensor, reward: torch.Tensor, model_output: Tuple):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_reward_output_mode() -> str:
        pass


class BaseMDNRNN(BaseRNN):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

        self.number_of_gaussians = model_parameters["number_of_gaussians"]
        self.use_gaussian_per_latent_dim: bool = model_parameters["use_gaussian_per_latent_dim"]

    def _predict_gaussian_mixture(self, model_output, temperature):
        # Temperature parameter is used at two points and only during sampling (not during training):
        # log_pi is divided with the temperature and sigma is multiplied with the square root of the temperature.
        # This is done in the reference implementation of the world model approach, specifically here:
        # https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py#L625 and on line 639

        # Shapes:
        # latents: (1, 1, L_SIZE)
        # mus, sigmas: (1, 1, N_GAUSS, L_SIZE)
        # log_pi: (1, 1, N_GAUSS, 1) or (1, 1, N_GAUSS, L_SIZE) depending on
        #         self.use_gaussian_per_latent_dim
        mus = model_output[0]
        sigmas = model_output[1]
        log_pi = model_output[2]

        number_of_gaussians = mus.size(2)
        latent_size = mus.size(3)

        # Remove (1, 1) dimensions and transpose to get (L_SIZE, N_GAUSS) dimensions
        mus = mus.view(number_of_gaussians, latent_size).t()
        sigmas = sigmas.view(number_of_gaussians, latent_size).t()

        if self.use_gaussian_per_latent_dim:
            log_pi = log_pi.view(number_of_gaussians, latent_size).t()
        else:
            log_pi = log_pi.view(number_of_gaussians, 1).t()

        # Exp-Normalization trick to calculate the softmax, useful to ensure numerical stability
        log_pi_temperature_adjusted = (log_pi / temperature)
        log_pi_temperature_adjusted -= log_pi_temperature_adjusted.max()
        log_pi_temperature_adjusted = log_pi_temperature_adjusted.exp()
        pi_temperature_adjusted = log_pi_temperature_adjusted / log_pi_temperature_adjusted.sum(dim=1, keepdim=True)

        # Randomly select a gaussian distribution either per dimension of the latent vector
        # (self.use_gaussian_per_latent_dim == True) or one gaussian distribution in general if
        # (self.use_gaussian_per_latent_dim == False)

        cumulated_sum = pi_temperature_adjusted.cumsum(dim=1)
        # Last entry is already 1.0 as the pis sum up to 1.0 because of the calculation above, but sometimes there
        # are slight rounding errors leading to results like 1.000001 or worse 0.9999994 etc. This leads to errors
        # in the drawing of the categorical distribution below, because if the random number is then 0.9999995
        # it would lead to a list of all False's which leads count_nonzero to count 0, which leads to getting an
        # index of num_gaussians - 0 = num_gaussians and this would be out of bounds!
        cumulated_sum[:, -1] = 1.0

        if self.use_gaussian_per_latent_dim:
            pi_count = torch.count_nonzero(
                cumulated_sum
                >= torch.rand((latent_size, 1), device=self.device),
                dim=1
            )
        else:
            # Take the cumulated sum of the pi's and then use a random number of the uniform distribution to do create
            # a categorical distribution. The first pi in the cumulated sum that is larger than the random number is
            # the sample drawn. To get the first number simply count all non zeros (i.e. larger pi's) and then subtract
            # from the size of pi at dim 1 (i.e. number_of_gaussians). Hopefully this is faster than using
            # torch.distributions.Categorical
            pi_count = torch.count_nonzero(cumulated_sum >= torch.rand(1, device=self.device))

        drawn_mixtures = number_of_gaussians - pi_count

        # Shape after this for selected_mus and selected_sigmas: (1, 1, L_SIZE)
        if self.use_gaussian_per_latent_dim:
            selected_mus = torch.gather(mus, dim=1, index=drawn_mixtures.unsqueeze(-1)).view(1, 1, latent_size)
            selected_sigmas = torch.gather(sigmas, dim=1, index=drawn_mixtures.unsqueeze(-1)).view(1, 1, latent_size)
        else:
            selected_mus = mus[:, drawn_mixtures].view(1, 1, latent_size)
            selected_sigmas = sigmas[:, drawn_mixtures].view(1, 1, latent_size)

        # Now use the randomly selected gaussian(s) to sample the next latent vector, i.e. the prediction
        random_vector = torch.randn(size=(1, 1, latent_size), device=self.device)
        latent_prediction = selected_mus + random_vector * selected_sigmas * torch.sqrt(temperature)

        return latent_prediction

    def predict(self, model_output, latents, temperature):
        latent_prediction = self._predict_gaussian_mixture(model_output, temperature)
        rewards = model_output[3]

        # latent_prediction: (BATCH_SIZE, SEQ_LEN, L_SIZE)
        return latent_prediction, self.denormalize_reward(rewards)

    def _forward_gaussian_mixture(self, gmm_outputs, sequence_length):
        stride = self.number_of_gaussians * self.latent_size

        if self.use_gaussian_per_latent_dim:
            mus, sigmas, pi, rewards = torch.split(
                gmm_outputs,
                split_size_or_sections=(stride, stride, stride, 1),
                dim=-1
            )
        else:
            mus, sigmas, pi, rewards = torch.split(
                gmm_outputs,
                split_size_or_sections=(stride, stride, self.number_of_gaussians, 1),
                dim=-1
            )

        mus = mus.view(self.batch_size, sequence_length, self.number_of_gaussians, self.latent_size)

        sigmas = sigmas.view(self.batch_size, sequence_length, self.number_of_gaussians, self.latent_size)
        sigmas = torch.exp(sigmas)

        if self.use_gaussian_per_latent_dim:
            pi = pi.view(self.batch_size, sequence_length, self.number_of_gaussians, self.latent_size)
        else:
            pi = pi.view(self.batch_size, sequence_length, self.number_of_gaussians, 1)

        # The pi's shall sum to one, therefore take the softmax over dimension 2 (number_of_gaussians)
        # Use log_softmax for numerical stability as we also compute NLLLoss directly in log-space
        log_pi = f.log_softmax(pi, dim=2)

        return mus, sigmas, log_pi, rewards

    def forward(self, latents: torch.Tensor, actions: torch.Tensor):
        sequence_length = latents.size(1)

        outputs, _ = self.rnn_forward(latents, actions)
        gmm_outputs = self.fc(outputs)

        mus, sigmas, log_pi, rewards = self._forward_gaussian_mixture(gmm_outputs, sequence_length)

        rewards = self.reward_output_activation_function(rewards)

        return mus, sigmas, log_pi, rewards

    @staticmethod
    def _predict_in_log_space(next_latent_vector, mus, sigmas, log_pi) -> torch.Tensor:
        # Natural logarithm applied on a Gaussian distribution
        log_prob = -torch.log(sigmas) - 0.5 * LOG2PI - 0.5 * torch.pow((next_latent_vector - mus) / sigmas, 2)

        # LogSumExp Trick for numerical stability, essentially this is the MDN formula
        # Dim 2 is number_of_gaussians
        log_prob_sum = torch.logsumexp(log_pi + log_prob, dim=2)

        return log_prob_sum

    def gmm_loss(self, next_latent_vector, mus, sigmas, log_pi):
        # next_latent_vector: (BATCH_SIZE, SEQ_LEN, L_SIZE)
        next_latent_vector = next_latent_vector.unsqueeze(2)

        log_prob_sum = self._predict_in_log_space(next_latent_vector, mus, sigmas, log_pi)

        nll = -log_prob_sum.mean()

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

    @staticmethod
    def get_reward_output_mode() -> str:
        return "mse"


class BaseSimpleRNN(BaseRNN):

    def __init__(self, model_parameters: dict, latent_size: int, batch_size: int, device: torch.device):
        super().__init__(model_parameters, latent_size, batch_size, device)

    def predict(self, model_output, latents=None, temperature=None):
        # This function mostly exists for mixture density network as the actual calculation of the next latent state
        # is not required for training, just the calculation of the predicted probability distribution.
        # But since we want to use the same interface, just return the prediction here
        return model_output[0], self.denormalize_reward(model_output[1])

    def forward(self, latents: torch.Tensor, actions: torch.Tensor):
        outputs, _ = self.rnn_forward(latents, actions)

        predictions = self.fc(outputs)

        predicted_latent_vector = predictions[:, :, :self.latent_size]
        predicted_reward = self.reward_output_activation_function(predictions[:, :, self.latent_size:])

        return predicted_latent_vector, predicted_reward

    def loss_function(self, next_latent_vector: torch.Tensor, reward: torch.Tensor, model_output: Tuple):
        predicted_latent, predicted_reward = model_output[0], model_output[1]

        # TODO check if reduction needs to be adapted to batch size and sequence length
        latent_loss = f.mse_loss(predicted_latent, next_latent_vector)
        reward_loss = f.mse_loss(predicted_reward, reward)

        loss = self.combine_latent_and_reward_loss(latent_loss=latent_loss, reward_loss=reward_loss)

        return loss, (latent_loss.item(), reward_loss.item())

    @staticmethod
    def get_reward_output_mode() -> str:
        return "mse"
