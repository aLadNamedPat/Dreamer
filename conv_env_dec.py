import torch
import torch.nn as nn
import torch.nn.functional as F



def nerf_posenc_2d(H, W, L=6, device='cpu'):
    """
    Returns a 2D NeRF-style positional encoding of shape (4L, H, W).
    - 2L channels for x (sin/cos)
    - 2L channels for y (sin/cos)
    """

    # Coordinates in [0, 1]
    x_lin = torch.linspace(0, 1, W, device=device)  # shape: (W,)
    y_lin = torch.linspace(0, 1, H, device=device)  # shape: (H,)

    # Broadcast to shape (H, W)
    x_coords = x_lin.unsqueeze(0).expand(H, W)      # (H, W)
    y_coords = y_lin.unsqueeze(1).expand(H, W)      # (H, W)

    # Build up lists of sin/cos expansions
    x_enc = []
    y_enc = []
    for i in range(L):
        freq = 2.0 ** i * torch.pi   # e.g. π, 2π, 4π, ...
        x_enc.append(torch.sin(freq * x_coords))
        x_enc.append(torch.cos(freq * x_coords))
        y_enc.append(torch.sin(freq * y_coords))
        y_enc.append(torch.cos(freq * y_coords))

    # Each list now has length 2L, each element shape (H, W).
    # Stack them on a new channel dimension => shape (2L, H, W)
    x_enc = torch.stack(x_enc, dim=0)
    y_enc = torch.stack(y_enc, dim=0)

    # Combine x and y encodings => shape (4L, H, W)
    pe_2d = torch.cat([x_enc, y_enc], dim=0)

    return pe_2d


class ConvEncoder(nn.Module):
    def __init__(self, depth=32, latent_size=32, act=nn.ReLU(), L=6):
        """
        depth: base channel multiplier
        latent_size: dimension of latent vector z
        act: activation function
        L: number of NeRF frequency bands
        """
        super().__init__()
        self.depth = depth
        self.act = act
        self.latent_size = latent_size
        self.L = L

        # The number of extra channels from NeRF encoding is 4L (2L for x, 2L for y).
        in_ch = 3 + 4 * L

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=1*depth, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=1*depth, out_channels=2*depth, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=2*depth, out_channels=4*depth, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=4*depth, out_channels=8*depth, kernel_size=4, stride=2)

        self.fc_mu = nn.Linear(8 * depth * 2 * 2, latent_size)
        self.fc_logvar = nn.Linear(8 * depth * 2 * 2, latent_size)

    def forward(self, obs):
        """
        obs shape: (B, T, H, W, 3). After permute => (B, T, 3, H, W).
        We'll flatten (B, T) => (B*T), add NeRF positional embeddings => (B*T, 3+4L, H, W).
        Then go through conv layers -> (B*T, ...). Finally split back to (B, T, latent_size).
        """
        # Permute to (B, T, C, H, W)
        obs = obs.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = obs.shape

        # Flatten batch
        x = obs.view(B*T, C, H, W)  # shape: (B*T, 3, H, W)

        # --------------------------------------------------
        #  Build multi-frequency positional encoding (NeRF)
        # --------------------------------------------------
        device = x.device
        # shape: (4L, H, W)
        pe_2d = nerf_posenc_2d(H, W, L=self.L, device=device)
        # Broadcast to match (B*T) in the batch dimension
        # but notice we only have (4L, H, W), not (B*T, 4L, H, W).
        # We can just repeat or expand along the batch dimension:
        pe_2d = pe_2d.unsqueeze(0).expand(B*T, -1, -1, -1)  # => (B*T, 4L, H, W)

        # Concatenate with original image channels => (B*T, 3 + 4L, H, W)
        x = torch.cat([x, pe_2d], dim=1)

        # CNN forward
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        # Flatten spatial dimensions
        x = x.reshape(x.size(0), -1)

        # Produce mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Reshape back to (B, T, latent_size)
        z = z.view(B, T, -1)
        print("Encoded shape:", z.shape)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, depth=32, latent_size=32, act=nn.ReLU(), shape=(64, 64, 3)):
        super().__init__()
        self.depth = depth
        self.latent_size = latent_size
        self.act = act
        self.out_height, self.out_width, self.out_channels = shape
        
        self.fc = nn.Linear(latent_size, 8 * depth * 1 * 1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=8 * depth, out_channels=8 * depth, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=8 * depth, out_channels=4 * depth, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=4 * depth, out_channels=2 * depth, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=2 * depth, out_channels=self.out_channels, kernel_size=6, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
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
