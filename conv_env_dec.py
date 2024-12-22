import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

def compute_encoder_output_size(input_shape, encoder):
    dummy_input = torch.zeros(input_shape).to(encoder.device)
    with torch.no_grad():
        output = encoder(dummy_input)
    return output.shape[-1]

class ConvEncoder(nn.Module):
    def __init__(self, depth=32, act=nn.ReLU()):
        super().__init__()
        self.depth = depth
        self.act = act
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1*depth, kernel_size=4, stride=2).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=1*depth, out_channels=2*depth, kernel_size=4, stride=2).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=2*depth, out_channels=4*depth, kernel_size=4, stride=2).to(self.device)
        self.conv4 = nn.Conv2d(in_channels=4*depth, out_channels=8*depth, kernel_size=4, stride=2).to(self.device)
        

    def forward(self, obs):
        obs = obs.to(self.device)
        obs = obs.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = obs.shape
        x = obs.view(B * T, C, H, W)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        x = x.view(B, T, -1)
        return x

class ConvDecoder(nn.Module):
    def calculate_flattened_size(self, channels, height, width):
        return channels * height * width
    

    def __init__(self, depth=32, act=nn.ReLU(), shape=(128, 192, 3)):
        super().__init__()
        self.depth = depth
        self.act = act
        self.out_height, self.out_width, self.out_channels = shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Linear(depth, 32 * depth).to(self.device)

        ### Hard Coded, should not be ###
        self.projection_layer = nn.Linear(12288, self.out_height * self.out_width * self.out_channels).to(self.device)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32*depth,
            out_channels=4*depth,
            kernel_size=5,
            stride=2
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=4*depth,
            out_channels=2*depth,
            kernel_size=5,
            stride=2
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=2*depth,
            out_channels=1*depth,
            kernel_size=6,
            stride=2
        )
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=1*depth,
            out_channels=self.out_channels,
            kernel_size=6,
            stride=2
        )

    def forward(self, features):
        features = features.to(self.device)
        B, T, F = features.shape
        features = features.view(B * T, F)
        
        x = self.fc(features)
        x = self.act(x)
        
        x = x.view(B * T, 32 * self.depth, 1, 1) 
        x = self.deconv1(x)  
        x = self.act(x)
        x = self.deconv2(x)  
        x = self.act(x)
        x = self.deconv3(x)  
        x = self.act(x)
        x = self.deconv4(x) 

        x = x.reshape(x.size(0), -1)
        x = self.projection_layer(x)
        x = x.view(B, T, self.out_height, self.out_width, self.out_channels)
        
        mean = x
        normal_dist = dist.Normal(loc=mean, scale=1.0)
        out_dist = dist.Independent(normal_dist, reinterpreted_batch_ndims=3)
        sample = out_dist.sample()
        return sample