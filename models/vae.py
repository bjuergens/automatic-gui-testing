import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 256 * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)  # 6
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2)  # 13
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)   # 27
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)  # 55
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)  # 111
        self.deconv6 = nn.ConvTranspose2d(32, self.img_channels, kernel_size=4, stride=2)  # 224

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.view(-1, 256, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        reconstruction = F.sigmoid(self.deconv6(x))
        return reconstruction


class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        # self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        # self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
