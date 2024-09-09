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

    def __init__(self, state_dim, action_dim, o_feature_dim, latent_dim, reward_dim):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(3, o_feature_dim)
        self.decoder = ConvDecoder(latent_dim * 2, 3)
        self.rnn = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        
        self.reward_model = RewardModel(latent_dim, state_dim, reward_dim)
        action_dim = action_dim.shape[0]
        self.transition_pre = nn.Linear(state_dim + action_dim, latent_dim)
        self.transition_post = nn.Linear(latent_dim, 2 * state_dim)
        
        self.representation_pre = nn.Linear(latent_dim + o_feature_dim, latent_dim)
        self.representation_post = nn.Linear(latent_dim, 2 * state_dim)
        self.relu = nn.ReLU()
    
    def forward(self, prev_state, actions, prev_belief, observations = None, nonterminals = None):
        encoded_observation = self.encoder(observations.float())
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, decoded_observations, rewards = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * (T - 1), [torch.empty(0)] * (T - 1)
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        for t in range(T- 1):
            _state = prior_states[t] if observations is None else posterior_states[t]
            print(f"State before nonterminals {_state}")
            _state = _state if (nonterminals is None or t == 0) else _state * nonterminals[t-1]
            hidden = self.relu(self.transition_pre(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            prior_means[t + 1], _prior_std_dev = torch.chunk(self.transition_post(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + 1e-5
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            print(f"Prior State Generated {prior_states[t+1]}")

            if observations is not None:
                hidden = self.relu(self.representation_pre(torch.cat([beliefs[t + 1][0], encoded_observation[t]], dim=0)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.representation_post(hidden), 2, dim=0)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + 1e-5
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
        
        new_prior_states = torch.zeros((T, self.latent_dim))
        new_posterior_states = torch.zeros((T, self.latent_dim))
        for i in range(len(prior_states)):
            new_prior_states[i] = prior_states[i].squeeze()
            new_posterior_states[i] = posterior_states[i].squeeze()
        decoded_observations = self.decoder(torch.cat((new_prior_states, new_posterior_states), dim = 1))
        hidden = [torch.stack(beliefs[1:], dim=0), new_prior_states[1:], torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [new_posterior_states[1:], torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0), decoded_observations]

        return hidden

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