import torch 
import torch.nn as nn
import torch.nn.functional as F
from conv_env_dec import ConvDecoder, ConvEncoder
import matplotlib.pyplot as plt

class RSSM(nn.Module):

    '''
    World Model Structure is the following
    Representation Model pθ(st | st-1, at-1, ot)
    Transition Model qθ(st | st-1, at-1)
    
    The following are descriptions of the params for intialization:
    state_dim --> the dimensions for the state 
    
    '''

    # TODO -- Add image feature dimension
    def __init__(self, state_dim, action_dim, o_feature_dim, o_dim, latent_dim, reward_dim):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.o_feature_dim = o_feature_dim
        self.reward_dim = reward_dim
        self.o_dim = o_dim
        self.o_feature_dim = o_feature_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.reward_dim = reward_dim

        self.encoder = ConvEncoder(self.o_feature_dim, self.latent_dim)
        self.decoder = ConvDecoder(self.o_feature_dim, latent_size=self.latent_dim * 2, shape=(o_dim[0], o_dim[1], 3))
        self.rnn = nn.GRUCell(input_size=self.latent_dim, hidden_size=self.latent_dim)
        
        self.reward_model = RewardModel(self.latent_dim, self.state_dim, self.reward_dim)
        self.action_dim = action_dim.shape[0]
        self.transition_pre = nn.Linear(self.state_dim + self.action_dim, self.latent_dim)
        self.transition_post = nn.Linear(self.latent_dim, 2 * self.state_dim)
        self.representation_pre = nn.Linear(self.latent_dim + self.o_feature_dim, self.latent_dim)
        self.representation_post = nn.Linear(self.latent_dim, 2 * self.state_dim)
        self.relu = nn.ReLU()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, prev_state, actions, prev_belief, observations = None, nonterminals = None):
        prev_state = prev_state.to(self.device)
        actions = actions.to(self.device)
        if observations is not None:
            observations = observations.to(self.device)
        # print(f"observations : {observations}")
        encoded_observation = self.encoder(observations.float())
        T = actions.size(1) + 1
        batch_size = actions.size(0)

        beliefs = torch.zeros(batch_size, T, self.latent_dim).to(self.device)
        prior_states = torch.zeros(batch_size, T, self.state_dim).to(self.device)
        prior_means = torch.zeros(batch_size, T, self.state_dim)
        prior_std_devs = torch.zeros(batch_size, T, self.state_dim)
        posterior_states = torch.zeros(batch_size, T, self.state_dim).to(self.device)
        posterior_means = torch.zeros(batch_size, T, self.state_dim)
        posterior_std_devs = torch.zeros(batch_size, T, self.state_dim)

        ## TODO Change Image Dimensions
        decoded_observations = torch.zeros(batch_size, T - 1, 3, 64, 64).to(self.device)
        rewards = torch.zeros(batch_size, T - 1, 1).to(self.device)
        
        # Why are these the same??
        beliefs[:, 0] = prev_belief.clone()
        prior_states[:, 0] = prev_state.clone()
        posterior_states[:, 0] = prev_state.clone()
        

        for t in range(T - 1):
            _state = (prior_states[:, t] if observations is None else posterior_states[:, t]).to(self.device)
            # print(f"State 1 Shape : {torch.cat([_state, actions[:, t]], dim=-1).shape}")
            # print(f"Actions 2 Shape : {torch.cat([_state, actions[:, t]], dim=1).shape}")
            # import pdb; pdb.set_trace()
            hidden = self.relu(self.transition_pre(torch.cat([_state, actions[:, t]], dim=-1)))

            new_belief = self.rnn(hidden, beliefs[:, t].clone())
            beliefs[:, t + 1] = new_belief

            prior_means[:, t + 1], _prior_std_dev = torch.chunk(self.transition_post(hidden), 2, dim=-1)
            prior_std_devs[:, t + 1] = F.softplus(_prior_std_dev) + 1e-5
            prior_states[:, t + 1] = prior_means[:, t + 1] + prior_std_devs[:, t + 1] * torch.randn_like(prior_means[:, t + 1])

            if observations is not None:
                beliefs_t = beliefs[:, t + 1]
                encoded_obs_t = encoded_observation[:, t]
                # print(f"Beliefs_t Shape: {beliefs_t.shape}")
                # print(f"Encoded_obs_t Shape: {encoded_obs_t.shape}")
                # print(f"Dim : {8 * self.o_feature_dim + self.latent_dim}")
                hidden = self.relu(self.representation_pre(torch.cat([beliefs_t, encoded_obs_t], dim=-1)))

            posterior_means[:, t + 1], _posterior_std_dev = torch.chunk(self.representation_post(hidden), 2, dim=1)
            posterior_std_devs[:, t + 1] = F.softplus(_posterior_std_dev) + 1e-5
            posterior_states[:, t + 1] = posterior_means[:, t + 1] + posterior_std_devs[:, t + 1] * torch.randn_like(posterior_means[:, t + 1])
            rewards[:, t] = self.reward_model(beliefs[:, t + 1], posterior_states[:, t + 1])
        # print(f"Prior States Shape: {prior_states.shape}")
        # print(f"Posterior States Shape: {posterior_states.shape}")
        # print(f"Concat Shape : {torch.cat((prior_states[:, 1:], posterior_states[:, 1:]), dim = -1).shape}")
        decoded_observations = self.decoder(
            torch.cat((prior_states[:, 1:], posterior_states[:, 1:]), dim=-1).to(self.device)
        )
        hidden = [beliefs[:, 1:], prior_states[:, 1:], prior_means[:, 1:], prior_std_devs[:, 1:]]
        if observations is not None:
            hidden += [posterior_states[:, 1:], posterior_means[:, 1:], posterior_std_devs[:, 1:], decoded_observations, rewards]
        
        # Plot the observation input and output
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # axes[0].imshow(observations[0, -1].cpu().detach())
        # axes[0].axis('off')
        # axes[0].set_title("Last Input Observation")
        
        # axes[1].imshow(decoded_observations[0, -1].cpu().detach())
        # axes[1].axis('off')
        # axes[1].set_title("Last Decoded Observation")
        
        # plt.show()
        return hidden

## Reward Model as defined by Reward Model qθ(rt | st):  
class RewardModel(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.hidden_dims = state_dim
        self.latent_dim = latent_dim
        ## Feedforward linear layers
        self.fw1 = nn.Linear(latent_dim + state_dim, hidden_dim)
        self.fw2 = nn.Linear(hidden_dim, hidden_dim)
        self.fw3 = nn.Linear(hidden_dim, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, latent_space, sampled_state):
        # Ensure both tensors are on the same device
        ## TODO : Remove this
        latent_space = latent_space.to(self.device)
        sampled_state = sampled_state.to(self.device)

        x = torch.cat([latent_space, sampled_state], dim=-1)
        x = self.relu(self.fw1(x))
        x = self.relu(self.fw2(x))
        reward = self.fw3(x)
        return reward