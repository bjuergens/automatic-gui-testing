import torch
import torch.nn as nn


class VAEFullInputSize(nn.Module):

    def __init__(self, latent_size: int):
        super().__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(self.latent_size)
        self.decoder = Decoder(self.latent_size)

    def encode(self, x):
        mu, log_var = self.encoder(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)

        return reconstruction

    def sample(self, number_of_samples: int):
        z = torch.randn((number_of_samples, self.latent_size))
        return self.decode(z)


class Encoder(nn.Module):

    def __init__(self, latent_size: int):
        super().__init__()

        self.latent_size = latent_size

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(2*2*256, self.latent_size)
        self.fc_log_var = nn.Linear(2*2*256, self.latent_size)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var


class Decoder(nn.Module):

    def __init__(self, latent_size: int):
        super().__init__()

        self.latent_size = latent_size

        # Input: (B x 224)
        # Reshape (B x

        self.fc = nn.Linear(self.latent_size, 2*2*256)

        # Input 2x2x256

        self.transposed_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=4, padding=1),
            nn.LeakyReLU()
        )

        self.transposed_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

        self.transposed_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

        self.transposed_conv_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=5, stride=4, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 2, 2)

        x = self.transposed_conv_1(x)
        x = self.transposed_conv_2(x)
        x = self.transposed_conv_3(x)
        x = self.transposed_conv_4(x)

        return x
