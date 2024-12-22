import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F


# class ConvEncoder(nn.Module):

#     def __init__(self, input_channels, feature_dim):
#         super(ConvEncoder, self).__init__()
#         self.relu = nn.ReLU()
        
#         ## We choose an enc of input channels -> 32 -> 64 -> 128 -> 256
#         # print("input channels: ", input_channels)
#         self.conv32 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv64 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv128 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv256 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.fc = nn.Linear(256 * 96, feature_dim)  

#     def forward(self, input):
#         # print(f"Conv Input {input}")
#         # print(f"Input shape: {input.shape}")
#         # input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])  
#         run = False
#         if len(input.shape) == 5:
#             run = True
#             data_point, length, width, height, channels = input.shape
#             input = input.view(-1, input.shape[2], input.shape[3], input.shape[4])
#         input = input.permute(0, 3, 1, 2)
#         x = self.relu(self.bn1(self.conv32(input)))
#         x = self.relu(self.bn2(self.conv64(x)))
#         x = self.relu(self.bn3(self.conv128(x)))
#         x = self.relu(self.bn4(self.conv256(x)))
#         ## Change dim to feature vector
#         if run:
#             x = x.reshape(data_point, length, -1)  
#         else:
#             x = x.reshape(x.size(0), -1)
#         x = self.fc(x)

#         return x
    
# class ConvDecoder(nn.Module):
#     def __init__(self, feature_dim, output_channels):
#         super(ConvDecoder, self).__init__()
#         self.relu = nn.ReLU()

#         ## ! Is sigmoid the correct activation function for this? 
#         # self.sigmoid = nn.Sigmoid()

#         ## Dec 256 -> 128 -> 64 -> 32 -> output_channels
#         self.fc = nn.Linear(feature_dim, 256 * 8 * 12)
#         self.deconv256 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.deconv128 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.deconv64 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.deconv32 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)

#     def forward(self, x):
#         print(f"x.shape {x.shape}")
#         x = self.fc(x)
#         x = x.reshape(x.size(0), 256, 8, 12)
#         x = self.relu(self.bn1(self.deconv256(x)))
#         x = self.relu(self.bn2(self.deconv128(x)))
#         x = self.relu(self.bn3(self.deconv64(x)))
#         x = self.relu(self.deconv32(x))
#         x = x.permute(0, 2, 3, 1)
#         return x

def compute_encoder_output_size(input_shape, encoder):
    dummy_input = torch.zeros(input_shape)
    with torch.no_grad():
        output = encoder(dummy_input)
    return output.shape[-1]

class ConvEncoder(nn.Module):
    def __init__(self, depth=32, act=nn.ReLU()):
        super().__init__()
        self.depth = depth
        self.act = act
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1*depth, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=1*depth, out_channels=2*depth, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=2*depth, out_channels=4*depth, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=4*depth, out_channels=8*depth, kernel_size=4, stride=2)
        

    def forward(self, obs):
        obs = obs.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = obs.shape
        x = obs.view(B * T, C, H, W)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        # print(f"Conv 4 Shape : {x.shape}")
        x = x.view(x.size(0), -1)
        x = x.view(B, T, -1)
        # print(f"Final Shape : {x.shape}")
        return x

class ConvDecoder(nn.Module):
    def calculate_flattened_size(self, channels, height, width):
        return channels * height * width
    

    def __init__(self, depth=32, act=nn.ReLU(), shape=(128, 192, 3)):
        super().__init__()
        self.depth = depth
        self.act = act
        self.out_height, self.out_width, self.out_channels = shape
        self.fc = nn.Linear(depth, 32 * depth)

        ### Hard Coded, should not be ###
        self.projection_layer = nn.Linear(12288, self.out_height * self.out_width * self.out_channels)
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
        B, T, F = features.shape
        features = features.view(B * T, F)
        # print(f"Features Shape : {features.shape}")
        
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
        # print(f"X Final Shape : {x.shape}")

        x = x.view(B * T, -1)
        x = self.projection_layer(x)
        x = x.view(B, T, self.out_height, self.out_width, self.out_channels)
        
        mean = x
        normal_dist = dist.Normal(loc=mean, scale=1.0)
        out_dist = dist.Independent(normal_dist, reinterpreted_batch_ndims=3)
        sample = out_dist.sample()
        # print(f"Sample Shape: {sample.shape}")
        return sample
        # print(f" WTFFF : {out_dist}")
        # return out_dist