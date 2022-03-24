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
        predicted_reward_in_logits = model_output[3]

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
        if self.use_gaussian_per_latent_dim:
            pi_count = torch.count_nonzero(
                pi_temperature_adjusted.cumsum(dim=1)
                >= torch.rand((pi_temperature_adjusted.size(0), 1), device=self.device),
                dim=1
            )
            drawn_mixtures = pi_temperature_adjusted.size(1) - pi_count
        else:
            # Take the cumulated sum of the pi's and then use a random number of the uniform distribution to do create
            # a categorical distribution. The first pi in the cumulated sum that is larger than the random number is
            # the sample drawn. To get the first number simply count all non zeros (i.e. larger pi's) and then subtract
            # from the size of pi at dim 1 (i.e. number_of_gaussians). Hopefully this is faster than using
            # torch.distributions.Categorical
            pi_count = torch.count_nonzero(pi_temperature_adjusted.cumsum(dim=1) >= torch.rand(1, device=self.device))
            drawn_mixtures = pi_temperature_adjusted.size(1) - pi_count

        # Shape after this for selected_mus and selected_sigmas: (1, 1, L_SIZE)
        if self.use_gaussian_per_latent_dim:
            selected_mus = torch.gather(mus, dim=1, index=drawn_mixtures.unsqueeze(-1)).view(latents.size())
            selected_sigmas = torch.gather(sigmas, dim=1, index=drawn_mixtures.unsqueeze(-1)).view(latents.size())
        else:
            selected_mus = mus[:, drawn_mixtures].view(latents.size())
            selected_sigmas = sigmas[:, drawn_mixtures].view(latents.size())

        # Now use the randomly selected gaussian(s) to sample the next latent vector, i.e. the prediction
        random_vector = torch.randn(size=latents.size(), device=self.device)
        latent_prediction = selected_mus + random_vector * selected_sigmas * torch.sqrt(temperature)

        # latent_prediction: (BATCH_SIZE, SEQ_LEN, L_SIZE)
        return latent_prediction, self.denormalize_reward(torch.sigmoid(predicted_reward_in_logits))

    def forward(self, latents: torch.Tensor, actions: torch.Tensor):
        sequence_length = latents.size(1)

        outputs, _ = self.rnn_forward(latents, actions)
        gmm_outputs = self.fc(outputs)

        stride = self.number_of_gaussians * self.latent_size

        # Don't apply activation function on rewards as the BCELossWithLogits applies a sigmoid internally.
        # Numerically more stable this way
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
