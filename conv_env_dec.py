import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F


## VAE Architecture comes form Ha's World Model Paper https://arxiv.org/pdf/1803.10122

class ConvEncoder(nn.Module):
    def __init__(self, depth=32, latent_size=32, act=nn.ReLU()):
        super().__init__()
        self.depth = depth
        self.act = act
        self.latent_size = latent_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1*depth, kernel_size=4, stride=2).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=1*depth, out_channels=2*depth, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=2*depth, out_channels=4*depth, kernel_size=4, stride=2).to(self.device)
        self.conv4 = nn.Conv2d(in_channels=4*depth, out_channels=8*depth, kernel_size=4, stride=2).to(self.device)
        self.fc_mu = nn.Linear(8*depth*2*2, latent_size).to(self.device)
        self.fc_logvar = nn.Linear(8*depth*2*2, latent_size).to(self.device)

    def forward(self, obs):
        obs = obs.to(self.device)
        # print(f"Obs Shape : {obs.shape}")
        obs = obs.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = obs.shape
        x = obs.view(B * T, C, H, W)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Latent Vector Z is sampled from Gaussian prior
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        z = z.view(B, T, -1)
        print("Shape of the encoded observation:", z.shape)
        return z

class ConvDecoder(nn.Module):
    def __init__(self, depth=32, latent_size=32, act=nn.ReLU(), shape=(64, 64, 3)):
        super().__init__()
        self.depth = depth
        self.latent_size = latent_size
        self.act = act
        self.out_height, self.out_width, self.out_channels = shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fc = nn.Linear(latent_size, 8 * depth * 1 * 1).to(self.device)
        self.deconv1 = nn.ConvTranspose2d(in_channels=8 * depth, out_channels=8 * depth, kernel_size=5, stride=2).to(self.device)
        self.deconv2 = nn.ConvTranspose2d(in_channels=8 * depth, out_channels=4 * depth, kernel_size=5, stride=2).to(self.device)
        self.deconv3 = nn.ConvTranspose2d(in_channels=4 * depth, out_channels=2 * depth, kernel_size=6, stride=2).to(self.device)
        self.deconv4 = nn.ConvTranspose2d(in_channels=2 * depth, out_channels=self.out_channels, kernel_size=6, stride=2).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        features = features.to(self.device)
        B, T, F = features.shape
        features = features.view(B * T, F)
        
        x = self.fc(features)
        x = self.act(x)
        
        x = x.reshape(B * T, 8 * self.depth, 1, 1)
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.act(self.deconv3(x))
        x = self.sigmoid(self.deconv4(x))
        
        x = x.reshape(B, T, self.out_height, self.out_width, self.out_channels)
        return x
