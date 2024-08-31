import torch
import torch.nn as nn
import numpy as np
from ReplayBuffer import Buffer 
from RSSM import RSSM
from torch.distributions.multivariate_normal import MultivariateNormal
import wandb

# device = torch.device("cuda")

class Dreamer(nn.Module):
    def __init__(
            self,
            env,
            state_dims : int,
            latent_dims : int,
            observation_dim : int,
            o_feature_dim : int,
            reward_dim : int,
            gamma : float  = 0.99,
            lambda_ : float = 0.95,
            batch_size : int = 50,
            batch_train_freq : int = 50,
            buffer_size : int = 100000000,
            sample_steps : int = 1000,
            ):
        super(Dreamer, self).__init__()
        
        self.env = env
        self.action_space = env.action_spec()
        self.state_dims = state_dims
        self.latent_dims = latent_dims
        self.observation_dim = observation_dim
        self.o_feature_dim = o_feature_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.batch_train_freq = batch_train_freq
        self.replayBuffer = Buffer(buffer_size)
        self.sample_steps = sample_steps

        # Actor needs to output the action to take at a standard deviation
        self.actor = DenseConnections(
            self.state_dims + self.latent_dims,
            self.action_space.shape[0],
            action_model = True
        )

        # Critic only needs to output the value of being at a certain latent dim (no sampling required)
        self.critic = DenseConnections(
            self.state_dims + self.latent_dims,
            1,
            action_model = False
        )

        # def __init__(self, state_dim, action_dim, observation_dim, o_feature_dim, latent_dim, reward_dim):
        self.RSSM = RSSM(
            state_dim=self.state_dims,
            action_dim=self.action_space,
            observation_dim=self.observation_dim,
            o_feature_dim=self.o_feature_dim,
            latent_dim=self.latent_dims,
            reward_dim=self.reward_dim
        )
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr =8e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr =8e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)

    # Sparkly fun things going on here
    def latent_imagine(self, latents, posterior, horizon : int):
    # Latent imagination receives the latents and the posterior where the latents are the probability distribution over possible events whereas the posterior is the deterministic

    # Posterior is a M x N vector representing the state at each different index
    # Latent is a M x N vector representing the latent at each different index
        x, y = posterior.shape

        imagined_state = posterior.reshape(x * y, -1)
        imagined_latent = latents.reshape(x * y, -1)

        action = self.actor(torch.cat([imagined_state, imagined_belief]).to(device=device))

        latent_list = [imagined_latent]
        state_list = [imagined_state]
        action_list = [action]

        for j in range(horizon):
            state = self.RSSM(imagined_state, action, imagined_belief)
            imagined_state, imagined_belief = state[0], state[1]
            action = self.actor(torch.cat([imagined_state, imagined_belief]).to(device=device))

            latent_list.append(imagined_latent)
            state_list.append(imagined_state)
            action_list.append(action_list)

        
        latent_list = torch.stack(latent_list, dim = 0).to(device = device)
        state_list = torch.stack(state_list, dim = 0).to(device = device)
        action_list = torch.stack(action_list, dim = 0).to(device = device)

        return latent_list, state_list, action_list

    # Will return new trajectories of states and actions that will be used to train our model

    def model_update(self):
        self.RSSM_optimizer = torch.optim.Adam(self.actor.parameters(), lr =8e-5)

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards_real, next_states, dones = self.replayBuffer.sample(self.batch_size, self.sample_steps, random_flag=True)
        
        # Get the initial state and latent space
        prev_state = torch.zeros((self.batch_size, self.RSSM.state_dim))
        prev_latent_space = torch.zeros((self.batch_size, self.RSSM.latent_dim))
        
        # Forward pass through the RSSM
        latent_spaces, prior_states, prior_means, prior_std_devs, \
        posterior_states, posterior_means, posterior_std_devs, \
        decoded_observations, rewards = self.RSSM(
            prev_state, 
            actions, 
            prev_latent_space, 
            nonterminals=1-dones, 
            observations=states
        )
        
        # Calculate the MSE loss for observation and decoded observation
        mse_loss = nn.MSELoss()
        observation_loss = mse_loss(states, decoded_observations)
        
        # Calculate the KL divergence loss between the prior and posterior distributions
        kl_loss = torch.distributions.kl_divergence(
            torch.distributions.Normal(posterior_means, posterior_std_devs),
            torch.distributions.Normal(prior_means, prior_std_devs)
        ).mean()

        beliefs, states, actions = self.latent_imagine(prev_state, posterior_means, 15)
        # TO DO: Calculate the following properly!!!!
        # Calculate the reward loss
        reward_loss = mse_loss(rewards_real, rewards)
        
        # Total loss
        total_loss = observation_loss + kl_loss + reward_loss
        
        # Backpropagation and optimization
        self.RSSM_optimizer.zero_grad()
        total_loss.backward()
        self.RSSM_optimizer.step()
        
        # Log losses to wandb
        wandb.log({
            "observation_loss": observation_loss.item(),
            "kl_loss": kl_loss.item(),
            "reward_loss": reward_loss.item(),
            "total_loss": total_loss.item()
        })
        
        return beliefs, states, actions


    # The agent is only training on the imagined states. All compute trajectories are imagined.
    def agent_update(
            self,
            beliefs,
            states,
        ):

        # Generates 50 random datapoints of length 50
        # This is going to have the reward of each state generated
        datapoints = torch.cat([beliefs, states], dim = 0)
        rewards = self.RSSM(datapoints.reshape(self.num_points * self.data_length, -1))[-1]
        rewards = rewards.reshape(self.num_points, self.data_length, -1)
        # This is going to have the value of each state generated, we want to flatten because the 
        values = self.critic(datapoints.reshape(self.num_points * self.data_length, -1))
        values = values.reshape(self.num_points, self.data_length, -1)

        # This should return the returns for each of the 50 randomly genearted trajectories
        returns = self.find_predicted_returns(
            rewards[:, :-1], # Remember that the batch_sample is two dimensional which means that the rewards and values will be two dimensional
            values[:, :-1],
            last_reward = rewards[:, -1],
            _lambda = self.lambda_
        )

        actor_loss = -torch.mean(self.find_predicted_returns()) #For actor loss it's enough to minimize the negative returns -> minimizing negative returns = maximizing positive returns
        self.actor_optimizer.zero_grad()
        actor_loss.backwards()
        self.actor_optimizer.step()

        critic_loss = -torch.mean(values.log_prob(returns))# For value loss (critic loss), we want to find the log probability of finding that returns for the given value predicted
        critic_loss.backwards()
        self.critic_optimizer.step()

        # Log losses to wandb
        wandb.log({
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        })

        # Use Log_prob as loss instead of MSE
        # Actor loss is the negative of the predicted returns
        # Value loss is the "KL" loss between the predicted value and the actual value 
        return actor_loss, critic_loss # Return the world model loss, actor loss, critic loss

    def rollout(
        self,
    ):
        t = 0
        while (t < self.batch_train_freq):
            t += 1
            self.num_timesteps += 1
            done = False
            action = self.sample_action(torch.cat([self.prev_state, self.prev_latent_space]))
            timestep = self.env.step(action)
            obs = torch.tensor(self.env.physics.render(camera_id=0, height=120, width=160))
            if (t == self.batch_train_freq):
                done = True
            latent_spaces, prior_states, prior_means, prior_std_devs, \
            posterior_states, posterior_means, posterior_std_devs, \
            decoded_observations, rewards = self.RSSM(
                self.prev_state, 
                action, 
                self.prev_latent_space, 
                nonterminals=1-done, 
                observations=obs
            )

            self.prev_state = posterior_states
            self.latent_space = latent_spaces

            self.replayBuffer.add((self.last_obs, action, timestep.reward, obs, done))
            self.last_obs = obs

    def train(
        self,
        timesteps : int,
        num_points : int,
        data_length : int,
    ):
        self.num_points = num_points
        self.data_length = data_length
        obs = self.env.reset()
        self.last_obs = torch.tensor(self.env.physics.render(camera_id=0, height=120, width=160)).to(device)
        self.prev_state = torch.zeros((self.batch_size, self.RSSM.state_dim))
        self.prev_latent_space = torch.zeros((self.batch_size, self.RSSM.latent_dim))

        self.num_timesteps = 0
        while (self.num_timesteps < timesteps):
            # wandb.init(project="dreamer_training", reinit=True)
            self.rollout()
            beliefs, states = self.model_update()
            # The data that the agent update receives should be the encoded space already to save memory
            actor_loss, critic_loss = self.agent_update(beliefs, states)
            obs = self.env.reset()

            # Log training progress to wandb
            wandb.log({
                "num_timesteps": self.num_timesteps,
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item()
            })

        return
    

    ### NEED TO EDIT THIS SO THAT REPRESENTATION MODEL ENCODES THE VALUES
    def sample_action(
        self,
        pixels : torch.Tensor,
        predict_mode : bool = False
    ) -> torch.Tensor:
        if (self.num_timesteps < self.sample_steps):
            return np.random.uniform(low=-1.0, high=1.0, size=self.env.action_spec().shape)
        elif not predict_mode:
            return self.actor(pixels) + 0.3 * torch.randn_like(self.env.action_spec().shape)
        else:
            return self.actor(pixels)


# Help from https://github.com/juliusfrost/dreamer-pytorch/blob/main/dreamer/algos/dreamer_algo.py for finding returns
# http://www.incompleteideas.net/book/RLbook2020.pdf

# Note that we are finding bootstrapped returns and not monte carlo returns at this step
    def find_predicted_returns(
        self,
        pred_rewards,
        pred_values,
        last_reward,
        _lambda
    ):
        # Need to first calculate the next values that since the pred_value are from the same state as the pred_rewards
        next_vals = pred_values[1:] # Might need to take the mean of this

        # Next, we need to calculate the predicted targets of the next states (This is just current_reward + (1 - lambda) * gamma * next_value)        
        targets = pred_rewards[:-1] + (1 - _lambda) * self.gamma * next_vals

        # Since we are using TD-lambda for finding the returns, this essentially correspond to the point that the returns on to 
        outputs = []
        curr_val = last_reward

        for i in range(len(pred_rewards) - 1, -1):
            curr_val = targets[i] + _lambda * self.gamma[i] * curr_val
            outputs.append(curr_val)

        return outputs


class DenseConnections(nn.Module):
    def __init__(self, 
                 input_dims : int, 
                 output_dims : int, 
                 mid_dims :int = 300, 
                 action_model : bool = False):
        super(DenseConnections, self).__init__()
        self.l1 = nn.Linear(input_dims, mid_dims)
        self.l2 = nn.Linear(mid_dims, mid_dims)
        self.l3 = nn.Linear(mid_dims, 2 * output_dims)

        self.action_model = action_model

    def forward(self, input : torch.Tensor):
        x = nn.ELU(self.l1(input))
        x = nn.ELU(self.l2(x))
        if not self.action_model: # For the value model
            mean, std = torch.chunk(self.l3(x), 2, dim=-1)
            cov_mat = torch.diagonal(std)
            return MultivariateNormal(mean, cov_mat)
        else: # For the actor model
            mean, std = torch.chunk(self.l3(x), 2, dim = -1)
            action = torch.tanh(mean + std * torch.randn_like(mean))
            return action
