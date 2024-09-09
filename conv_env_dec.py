import torch
import torch.nn as nn

class ConvEncoder(nn.Module):

    def __init__(self, input_channels, feature_dim):
        super(ConvEncoder, self).__init__()
        self.relu = nn.ReLU()
        
        ## We choose an enc of input channels -> 32 -> 64 -> 128 -> 256
        # print("input channels: ", input_channels)
        self.conv32 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv64 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv128 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv256 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 96, feature_dim)  

    def forward(self, input):
        # print(f"Conv Input {input}")
        # print(f"Input shape: {input.shape}")
        # input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])  
        run = False
        if len(input.shape) == 5:
            run = True
            data_point, length, width, height, channels = input.shape
            input = input.view(-1, input.shape[2], input.shape[3], input.shape[4])
        input = input.permute(0, 3, 1, 2)
        x = self.relu(self.bn1(self.conv32(input)))
        x = self.relu(self.bn2(self.conv64(x)))
        x = self.relu(self.bn3(self.conv128(x)))
        x = self.relu(self.bn4(self.conv256(x)))
        ## Change dim to feature vector
        if run:
            x = x.reshape(data_point, length, -1)  
        else:
            x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
    
class ConvDecoder(nn.Module):
    def __init__(self, feature_dim, output_channels):
        super(ConvDecoder, self).__init__()
        self.relu = nn.ReLU()

        ## ! Is sigmoid the correct activation function for this? 
        # self.sigmoid = nn.Sigmoid()

        ## Dec 256 -> 128 -> 64 -> 32 -> output_channels
        self.fc = nn.Linear(feature_dim, 256 * 8 * 12)
        self.deconv256 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv128 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv64 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv32 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.size(0), 256, 8, 12)
        x = self.relu(self.bn1(self.deconv256(x)))
        x = self.relu(self.bn2(self.deconv128(x)))
        x = self.relu(self.bn3(self.deconv64(x)))
        x = self.relu(self.deconv32(x))
        x = x.permute(0, 2, 3, 1)
        return x
