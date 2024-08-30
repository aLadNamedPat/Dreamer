import torch
import torch.nn as nn
import numpy as np
from ReplayBuffer import Buffer 
from RSSM import RSSM
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cuda")

class Dreamer(nn.Module):
    def __init__(
            self,
            env,
            input_dims : int,
            output_dims : int, 
            action_space : int,
            gamma : float  = 0.99,
            lambda_ : float = 0.95,
            batch_size : int = 50,
            batch_train_freq : int = 50,
            buffer_size : int = 1e8,
            sample_steps : int = 1000,
            ):
        super(Dreamer, self).__init__()

        self.world_model = nn.Sequential(
            # Fill in something here for a model the Sequential is just a stand-in
        )

        
        self.env = env
        self.action_space = env.action_space
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space
        self.gamma = gamma
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.batch_train_freq = batch_train_freq
        self.replayBuffer = Buffer(buffer_size)

        self.actor = DenseConnections(
            input_dims,
            action_space,
            action_model = True
        )

        self.critic = DenseConnections(
            input_dims,
            output_dims,
            action_model = False
        )

        # RANDOM REWARD MODEL INCLUDED FOR NOW
        self.rewards = DenseConnections(
            input_dims,
            output_dims,
            action_model = False
        )


    def update(
            self,
            states : torch.Tensor,
            ):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr =8e-5)
        self.batch_sample = self.replayBuffer.sample()
        rewards = self.rewards(states)
        values = self.critic(states)

        returns = self.find_predicted_returns(
            rewards[:-1],
            values[:-1],
            last_reward = rewards[-1],
            _lambda = self.lambda_
        )

        actor_loss = -torch.mean(self.find_predicted_returns()) #For actor loss it's enough to minimize the negative returns -> minimizing negative returns = maximizing positive returns
        self.actor_optimizer.zero_grad()
        actor_loss.backwards()
        self.actor_optimizer.step()

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)
        critic_distribution  = self.critic(self.batch_sample)

        critic_loss = -torch.mean(values.log_prob(returns))# For value loss (critic loss), we want to find the log probability of finding that returns for the given value predicte

        # Use Log_prob as loss instead of MSE
        # Actor loss is the negative of the predicted returns
        # Value loss is the "KL" loss between the predicted value and the actual value 
        return # Return the world model loss, actor loss, critic loss
    
    def rollout(
        self,
    ):
        t = 0

        while (t < self.batch_train_freq):
            t += 1
            self.num_timesteps += 1
            done = False
            action = self.sample_action(self.last_obs.to(device))
            timestep = self.env.step(action)
            obs = torch.tensor(self.env.physics.render(camera_id=0, height=120, width=160))
            if (t == self.batch_train_freq):
                done = True
            self.replayBuffer.add((self._last_obs, action, timestep.reward, obs, done))
            self.last_obs = obs


    def train(
        self,
        timesteps : int
    ):
        obs = self.env.reset()
        self._last_obs = torch.tensor(self.env.physics.render(camera_id=0, height=120, width=160)).to(device)
        self.num_timesteps = 0
        while (self.num_timesteps < timesteps):
            self.rollout()
            data_points = self.replayBuffer.sample()
            self.update(data_points)
        
    def sample_action(
        self,
        pixels : torch.Tensor,
    ) -> torch.Tensor:
        if (self.num_timesteps < self.sample_steps):
            return np.random.uniform(low=-1.0, high=1.0, size=self.env.action_spec().shape)
        else:
            return self.actor(pixels).sample()
    
    
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
                 predict_std : bool = False):
        super(DenseConnections, self).__init__()
        self.l1 = nn.Linear(input_dims, mid_dims)
        self.l2 = nn.Linear(mid_dims, mid_dims)

        self.predict_std = predict_std
        if self.predict_std:
            self.std = nn.Linear(mid_dims, output_dims)
            self.mean = nn.Linear(mid_dims, output_dims)
        else:
            self.l3 = nn.Linear(mid_dims, output_dims)


    def forward(self, input : torch.Tensor):
        x = nn.ELU(self.l1(input))
        x = nn.ELU(self.l2(x))
        if self.predict_std:
            mean = self.mean(x)
            std = self.std(x)
            cov_mat = torch.diagonal(std)
            return MultivariateNormal(mean, cov_mat)
        else:
            x = self.l3(x)

        return x
    