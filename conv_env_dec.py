import torch
import torch.nn as nn

class ConvEncoder(nn.Module):

    def __init__(self, input_channels, feature_dim):
        super(ConvEncoder, self).__init__()
        self.relu = nn.ReLU()
        
        ## We choose an enc of input channels -> 32 -> 64 -> 128 -> 256
        self.conv32 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv64 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv128 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv256 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(256 * 4 * 4, feature_dim)

    def forward(self, input):
        x = self.relu(self.conv32(input))
        x = self.relu(self.conv64(x))
        x = self.relu(self.conv128(x))
        x = self.relu(self.conv256(x))
        ## Change dim to feature vector
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, feature_dim, output_channels):
        super(ConvDecoder, self).__init__()
        self.relu = nn.ReLU()

        ## ! Is sigmoid the correct activation function for this? 
        self.sigmoid = nn.Sigmoid()

        ## Dec 256 -> 128 -> 64 -> 32 -> output_channels
        self.fc = nn.Linear(feature_dim, 256 * 4 * 4)
        self.deconv256 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv128 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv64 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv32 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 256, 4, 4)
        x = self.relu(self.deconv256(x))
        x = self.relu(self.deconv128(x))
        x = self.relu(self.deconv64(x))
        x = self.sigmoid(self.deconv32(x))
        return x
