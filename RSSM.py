import torch 
import torch.nn as nn
import torch.nn.functional as F
from conv_env_dec import ConvDecoder, ConvEncoder

class RSSM(nn.Module):
    '''
    World Model Structure is the following
    Representation Model pθ(st | st-1, at-1, ot)
    Transition Model qθ(st | st-1, at-1): 
    Reward Model qθ(rt | st):  
    '''

    def __init__(self, state_dim, action_dim, observation_dim, o_feature_dim, hidden_dim, reward_dim):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.encoder = ConvEncoder(observation_dim, o_feature_dim)
        self.decoder = ConvDecoder(hidden_dim, observation_dim)
        self.rnn = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        
        self.transition_pre = nn.Linear(state_dim + action_dim, hidden_dim)
        self.transition_post = nn.Linear(hidden_dim, 2 * state_dim)
        
        self.representation_pre = nn.Linear(hidden_dim + o_feature_dim, hidden_dim)
        self.representation_post = nn.Linear(hidden_dim, 2 * state_dim)
        self.relu = nn.ReLU()
    
    ## Heavily based off of: https://github.com/zhaoyi11/dreamer-pytorch/blob/master/models.py 
    def forward(self, prev_state, actions, prev_belief, nonterminals = None, observations = None):
        time_steps = actions.shape[0] + 1
        beliefs = [torch.empty(0)] * time_steps
        prior_states = [torch.empty(0)] * time_steps
        prior_means = [torch.empty(0)] * time_steps
        prior_std_devs = [torch.empty(0)] * time_steps
        posterior_states = [torch.empty(0)] * time_steps
        posterior_means = [torch.empty(0)] * time_steps
        posterior_std_devs = [torch.empty(0)] * time_steps
        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state

        
        for t in range(time_steps - 1):
            if observations is None:
                state = prior_states[t]
            else:
                state = posterior_states[t]

            if nonterminals and t != 0:
                state = state * nonterminals[t-1] 
            
            beliefs[t+1] = self.rnn(hidden, beliefs[t])
            
            hidden = self.relu(self.transition_pre(torch.cat([state, actions[t]], dim = 1)))
            prior_means[t+1], _prior_std_dev = torch.chunk(self.transition_post(hidden), 2, dim = 1)
            prior_std_devs[t+1] = F.softplus(_prior_std_dev)
            cov_matrix = torch.diag_embed(prior_std_devs[t+1]**2)
            latent_state = torch.distributions.MultivariateNormal(prior_means[t+1], cov_matrix).rsample()
            prior_states[t+1] = latent_state

            if observations is not None:
                encoded_observation = self.encoder(observations[t])
                hidden = self.relu(self.representation_pre(torch.cat([beliefs[t+1], encoded_observation], dim=1)))
                posterior_means[t+1], _posterior_std_dev = torch.chunk(self.representation_post(hidden), 2, dim=1)
                posterior_std_devs[t+1] = F.softplus(_posterior_std_dev)
                cov_matrix = torch.diag_embed(posterior_std_devs[t+1]**2)
                latent_state = torch.distributions.MultivariateNormal(posterior_means[t+1], cov_matrix).rsample()
                posterior_states[t+1] = latent_state
            
            ## Returns the beliefs, states, means, and standard deviations
            states = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
            if observations:
                states += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
                decoded_observations = [self.decoder(state) for state in posterior_states[1:]]
                states.append(torch.stack(decoded_observations, dim=0))
                
            return states