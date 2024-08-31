import torch 
import torch.nn as nn
import torch.nn.functional as F
from conv_env_dec import ConvDecoder, ConvEncoder

class RSSM(nn.Module):
    '''
    World Model Structure is the following
        Representation Model pθ(st | st-1, at-1, ot):
            Encoder (observation_dim, latent_dim) --> MLP (latent_dim + action_dim + latent_dim, hidden_dim) --> GRU Block 
            --> Decoder(hidden_dim) 
        
        Transition Model qθ(st | st-1, at-1): 
            MLP (latent_dim + action_dim, hidden_dim) --> GRU Block 

        Reward Model qθ(rt | st):  
            
    '''

    def __init__(self, latent_dim, action_dim, observation_dim, hidden_dim):
        super(RSSM, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = ConvEncoder(observation_dim, latent_dim)
        self.decoder = ConvDecoder(hidden_dim, observation_dim)
        self.rnn = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        self.representation_prefix = nn.Linear(2 * latent_dim + action_dim, hidden_dim),
        self.representation_post = nn.Linear(hidden_dim, 2 * latent_dim)

        self.transition_prefix = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.transition_post = nn.Linear(hidden_dim, 2 * latent_dim)

        self.reward_prefix = nn.Linear(latent_dim, hidden_dim)
        self.reward_post = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
    
    def representation(self, prev_state, prev_action, observation):
        input = torch.cat([prev_state, prev_action, observation], dim=-1)
        output = self.representation_model(input)
        mean, std = torch.chunk(output, 2, dim=-1)
        return mean, std
    
    def transition(self, prev_state, prev_action):
        input = torch.cat([prev_state, prev_action], dim=-1)
        output = self.transition_model(input)
        mean, std = torch.chunk(output, 2, dim=-1)
        return mean, std
    
    def reward(self, latent_state):
        reward = self.reward_model(latent_state)
        return reward
    
    def forward(self, prev_state, prev_action, observation):
        mean, std = self.representation(prev_state, prev_action, observation)
        std = F.softplus(std)  # Ensure std is positive
        cov_matrix = torch.diag_embed(std**2)
        latent_state = torch.distributions.MultivariateNormal(mean, cov_matrix).rsample()
        belief = self.rnn(torch.cat([latent_state, prev_action], dim=-1), prev_state)
        
        return {
            'mean': mean,
            'std': std,
            'sample': latent_state,
            'belief': belief,
            'rnn_state': belief,
        }