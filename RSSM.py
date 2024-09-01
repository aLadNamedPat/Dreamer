import torch 
import torch.nn as nn
import torch.nn.functional as F
from conv_env_dec import ConvDecoder, ConvEncoder

class RSSM(nn.Module):
    '''
    World Model Structure is the following
    Representation Model pθ(st | st-1, at-1, ot)
    Transition Model qθ(st | st-1, at-1)
    
    The following are descriptions of the params for intialization:
    state_dim --> the dimensions for the state 
    
    '''
    def __init__(self, state_dim, action_dim, observation_dim, o_feature_dim, latent_dim, reward_dim):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(3, o_feature_dim)
        self.decoder = ConvDecoder(latent_dim, 3)
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        
        self.reward_model = RewardModel(latent_dim, state_dim, reward_dim)
        action_dim = action_dim.shape[0]
        self.transition_pre = nn.Linear(state_dim + action_dim, latent_dim)
        self.transition_post = nn.Linear(latent_dim, 2 * state_dim)
        
        self.representation_pre = nn.Linear(latent_dim + o_feature_dim, latent_dim)
        self.representation_post = nn.Linear(latent_dim, 2 * state_dim)
        self.relu = nn.ReLU()
    
    ## Heavily based off of: https://github.com/zhaoyi11/dreamer-pytorch/blob/master/models.py 
    def forward(self, prev_state, action, prev_latent_space, nonterminals = None, observation = None): 
        latent_space = prev_latent_space
        prior_state = prev_state
        posterior_state = prev_state

        if observation is None:
            state = prior_state
        else:
            state = posterior_state

        if nonterminals is not None:
            state = state * nonterminals
            
        latent_space = self.rnn(state.view(-1, state.size(-1)), latent_space.view(-1, latent_space.size(-1)))
        
        ## TODO : Do we need to reduce the size of the state
        
        # state = state.view(-1)
        hidden = self.relu(self.transition_pre(torch.cat([state, action], dim = -1)))
        prior_mean, _prior_std_dev = torch.chunk(self.transition_post(hidden), 2, dim = -1)
        prior_std_dev = F.softplus(_prior_std_dev) + 1e-5
        cov_matrix = torch.diag_embed(prior_std_dev**2)
        sampled_state = torch.distributions.MultivariateNormal(prior_mean, cov_matrix)
        prior_state = sampled_state.rsample()

        if observation is not None:
            observation = observation.float()
            # print(f"Observation Shape: {observation.shape}")
            encoded_observation = self.encoder(observation)
            # print(f"Latent Space Shape: {latent_space.shape}")
            # print(f"Encoded Observation Shape: {encoded_observation.shape}")
            hidden = self.relu(self.representation_pre(torch.cat([latent_space, encoded_observation], dim=-1)))
            posterior_mean, _posterior_std_dev = torch.chunk(self.representation_post(hidden), 2, dim=-1)
            posterior_std_dev = F.softplus(_posterior_std_dev) + 1e-5
            cov_matrix = torch.diag_embed(posterior_std_dev**2)
            sampled_state = torch.distributions.MultivariateNormal(posterior_mean, cov_matrix)
            posterior_state = sampled_state.rsample()
            
        reward = self.reward_model(latent_space, posterior_state)
        
        states = [latent_space, prior_state, prior_mean, prior_std_dev]
        if observation is not None:
            states = states + [posterior_state, posterior_mean, posterior_std_dev]
            decoded_observation = self.decoder(posterior_state)
            states.append(decoded_observation)
                
        states.append(reward)
        return states

## Reward Model as defined by Reward Model qθ(rt | st):  
class RewardModel(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_dim):
        super().__init__()
        self.relu = nn.ReLU()
        ## Feedforward linear layers
        self.fw1 = nn.Linear(latent_dim + state_dim, hidden_dim)
        self.fw2 = nn.Linear(hidden_dim, hidden_dim)
        self.fw3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, latent_space, sampled_state):
        latent_space = latent_space.reshape(sampled_state.shape)
        # print(latent_space.shape)
        # print(sampled_state.shape)
        x = torch.cat([latent_space, sampled_state], dim=-1)
        x = self.relu(self.fw1(x))
        x = self.relu(self.fw2(x))
        reward = self.fw3(x)
        return reward
