"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import Parameter

ONE_OVER_SQRT_2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)


def _gmm_loss_using_log(batch, mus, sigmas, log_pi, reduce=True):
    log_prob = -torch.log(sigmas) - 0.5 * LOG2PI - 0.5 * torch.pow((batch - mus) / sigmas, 2)
    log_prob_sum = log_prob.sum(dim=-1)
    log_prob_sum = torch.logsumexp(log_pi + log_prob_sum, dim=-1)

    if reduce:
        nll = -log_prob_sum.mean()
    else:
        nll = -log_prob_sum

    return nll


def _gmm_loss(batch, mus, sigmas, log_pi, reduce=True):
    prob = ONE_OVER_SQRT_2PI * torch.exp(-0.5 * torch.pow((batch - mus) / sigmas, 2)) / sigmas
    log_prob = torch.log(prob).sum(dim=-1)
    log_prob = torch.logsumexp(log_pi + log_prob, dim=-1)

    if reduce:
        nll = -log_prob.mean()
    else:
        nll = -log_prob

    return nll


def gmm_loss(batch, mus, sigmas, log_pi, reduce=True):
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args log_pi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)

    # nll_using_log = _gmm_loss_using_log(batch, mus, sigmas, log_pi, reduce)
    nll = _gmm_loss(batch, mus, sigmas, log_pi, reduce)

    # assert torch.isclose(nll_using_log, nll)

    return nll


class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians, batch_size, device):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.batch_size = batch_size
        self.device = device

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 1)

    def forward(self, *inputs):
        pass


class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians, batch_size, device):
        super().__init__(latents, actions, hiddens, gaussians, batch_size, device)
        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.hidden_state, self.cell_state = self.initialize_hidden()

    def initialize_hidden(self):
        hidden = torch.zeros((self.rnn.num_layers, self.batch_size, self.hiddens), device=self.device)
        cell = torch.zeros((self.rnn.num_layers, self.batch_size, self.hiddens), device=self.device)
        return (hidden, cell)

    def forward(self, actions, latents):
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        # outs, (hz, cz) = self.rnn(ins)
        outs, (self.hidden_state, self.cell_state) = self.rnn(ins, (self.hidden_state, self.cell_state))
        self.hidden_state, self.cell_state = self.hidden_state.detach(), self.cell_state.detach()

        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        log_pi = f.log_softmax(pi, dim=-1)

        rewards = gmm_outs[:, :, -1]

        return mus, sigmas, log_pi, rewards


class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden):
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
