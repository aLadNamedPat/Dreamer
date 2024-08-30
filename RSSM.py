import torch 
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, observation_dim, hidden_dim, latent_dim):
        super(RSSM, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.GRUCell(input_size=latent_dim + action_dim, hidden_size=hidden_dim)
        
        # Representation model pθ(st | st-1, at-1, ot)
        self.representation_model = nn.Sequential(
            nn.Linear(state_dim + action_dim + observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        
        # Transition model qθ(st | st-1, at-1)
        self.transition_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        
        # Reward model qθ(rt | st)
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
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
        latent_state = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std)).rsample()
        belief = self.rnn(torch.cat([latent_state, prev_action], dim=-1), prev_state)
        
        return {
            'mean': mean,
            'std': std,
            'sample': latent_state,
            'belief': belief,
            'rnn_state': belief,
        }
