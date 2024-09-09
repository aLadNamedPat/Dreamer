import torch
import torch.nn as nn
import numpy as np
from ReplayBuffer import Buffer 
from RSSM import RSSM
from torch.distributions.multivariate_normal import MultivariateNormal
import wandb
import pickle
import gzip
import torch.nn.functional as F

device = torch.device("cpu")

wandb.init(
    project="Dreamer",
    config={
    "learning_rate": 0.001,
    # Add other hyperparameters here
    },
    reinit=True,
)

class Dreamer(nn.Module):
    def __init__(
            self,
            env,
            state_dims : int,
            latent_dims : int,
            o_feature_dim : int,
            reward_dim : int,
            gamma : float  = 0.99,
            lambda_ : float = 0.95,
            batch_size : int = 50,
            batch_train_freq : int = 50,
            buffer_size : int = 100000000,
            sample_steps : int = 50,
            steps_of_sampling : int = 1000,
            horizon : int = 15,
            ):
        super(Dreamer, self).__init__()
        
        self.env = env
        self.action_space = env.action_spec()
        self.state_dims = state_dims
        self.latent_dims = latent_dims
        self.o_feature_dim = o_feature_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.batch_train_freq = batch_train_freq
        self.replayBuffer = Buffer(buffer_size)
        self.sample_steps = sample_steps
        self.steps_of_sampling = steps_of_sampling
        self.horizon = horizon

        # Actor needs to output the action to take at a standard deviation
        self.actor = DenseConnections(
            self.state_dims + self.latent_dims,
            self.action_space.shape[0],
            action_model = True
        ).to(device)

        # Critic only needs to output the value of being at a certain latent dim (no sampling required)
        self.critic = DenseConnections(
            self.state_dims + self.latent_dims,
            1,
            action_model = False
        ).to(device)

        # def __init__(self, state_dim, action_dim, observation_dim, o_feature_dim, latent_dim, reward_dim):
        self.RSSM = RSSM(
            state_dim=self.state_dims,
            action_dim=self.action_space,
            o_feature_dim=self.o_feature_dim,
            latent_dim=self.latent_dims,
            reward_dim=self.reward_dim
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr =8e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)
        self.RSSM_optimizer = torch.optim.Adam(self.RSSM.parameters(), lr =6e-4)

    # Sparkly fun things going on here
    def latent_imagine(self, latents, posterior, horizon : int):
    # Latent imagination receives the latents and the posterior where the latents are the probability distribution over possible events whereas the posterior is the deterministic

    # Posterior is a M x N vector representing the state at each different index
    # Latent is a M x N vector representing the latent at each different index
        x, y = posterior.shape

        # imagined_state = posterior.reshape(x * y, -1)
        # imagined_latent = latents.reshape(x * y, -1)
        imagined_state = posterior
        imagined_latent = latents
        action = self.actor(torch.cat([imagined_state, imagined_latent], -1))
        # print(f"Action Reshape {action.reshape(x, y, -1)}")

        latent_list = [imagined_latent]
        state_list = [imagined_state]
        action_list = [action]

        for _ in range(horizon):
            state = self.RSSM(imagined_state, action_list, imagined_latent)
            imagined_state, imagined_latent = state[0], state[1]
            action = self.actor(torch.cat([imagined_state, imagined_latent], -1))
            # action.reshape(x, y, -1)
            latent_list.append(imagined_latent)
            state_list.append(imagined_state)
            action_list.append(action)
        
        latent_list = torch.stack(latent_list, dim = 0)
        state_list = torch.stack(state_list, dim = 0)
        action_list = torch.stack(action_list, dim = 0)

        return latent_list, state_list, action_list

    # Will return new trajectories of states and actions that will be used to train our model

    def model_update(self):

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards_real, next_states, dones = self.replayBuffer.sample(self.batch_size, self.sample_steps)
        print(states.shape)
        print(actions.shape)
        print(rewards_real.shape)
        print(dones.shape)

        # Get the initial state and latent space

        prev_state = torch.zeros((self.batch_size, self.RSSM.state_dim))
        prev_latent_space = torch.zeros((self.batch_size, self.RSSM.latent_dim))
        # Forward pass through the RSSM
        # print(f"Dones: {dones}")
        # print(f"actions: {actions.squeeze()}")
        # print(f"states: {prev_state.shape}")

        latent_spaces, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, decoded_observations, rewards = self.RSSM(
            prev_state.to(device),
            actions.squeeze().float().to(device),
            prev_latent_space.to(device), 
            nonterminals=torch.logical_not(dones).to(device), 
            observations=states.to(device)
        )
        
        # Calculate the MSE loss for observation and decoded observation
        mse_loss = nn.MSELoss()
        observation_loss = mse_loss(states.float().to(device), decoded_observations)
        
        # Calculate the KL divergence loss between the prior and posterior distributions
        kl_loss = torch.distributions.kl_divergence(
            torch.distributions.Normal(posterior_means, posterior_std_devs),
            torch.distributions.Normal(prior_means, prior_std_devs)
        ).mean()
        
        beliefs, states, actions = self.latent_imagine(prev_state.to(device), posterior_means.to(device), self.horizon)
        ## TODO: Calculate the following properly!!!!
        # Calculate the reward loss
        
        reward_loss = mse_loss(rewards_real.float().to(device), rewards.squeeze()).float()
        # Total loss
        total_loss = observation_loss + kl_loss + reward_loss

        # Backpropagation and optimization
        self.RSSM_optimizer.zero_grad()
        total_loss.backward()
        self.RSSM_optimizer.step()
        
        # Log losses to wandb
        # wandb.log({
        #     "observation_loss": observation_loss.item(),
        #     "kl_loss": kl_loss.item(),
        #     "reward_loss": reward_loss.item(),
        #     "total_loss": total_loss.item()
        # })
        
        return beliefs, states, actions, reward_loss, kl_loss, observation_loss


    # The agent is only training on the imagined states. All compute trajectories are imagined.
    def agent_update(
            self,
            beliefs,
            states,
            actions,
        ):

        # Generates 50 random datapoints of length 50
        # This is going to have the reward of each state generated
        rewards = self.RSSM(states, actions, beliefs)[-1]
        # rewards = rewards.reshape(self.num_points, self.data_length, -1)
        # This is going to have the value of each state generated, we want to flatten because the 
        # print(f'beliefs: {beliefs.shape}')
        # print(f'states: {states.shape}')
        values = self.critic(torch.cat([states, beliefs], dim = -1).detach())
        # values = values.reshape(self.num_points, self.data_length, -1)

        # This should return the returns for each of the 50 randomly genearted trajectories

        discounts = self.gamma * torch.ones_like(torch.cat([states, beliefs], dim = -1).detach())
        discount_arr = torch.cat([torch.ones_like(discounts[:1]), discounts[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        
        # print(f"reward: {rewards.shape}")
        # print(f"values: {values}")
        # returns = self.find_predicted_returns(
        #     rewards[:, :-1], # Remember that the batch_sample is two dimensional which means that the rewards and values will be two dimensional
        #     values.mean[:, :-1],
        #     last_reward = rewards[:, -1],
        #     _lambda = self.lambda_
        # )

        returns = self.find_predicted_returns(
            rewards[-1], # Remember that the batch_sample is two dimensional which means that the rewards and values will be two dimensional
            values.mean[-1],
            last_reward = rewards[-1],
            _lambda = self.lambda_
        )
        
        actor_loss = -torch.mean(discount * returns)
        # print(f"returns: {returns}")
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # with torch.no_grad():
        #     values = self.critic(torch.cat([states, beliefs], dim = -1)[:,:-1])
        
        critic_loss = -torch.mean(values.log_prob(returns))# For value loss (critic loss), we want to find the log probability of finding that returns for the given value predicted
        critic_loss.backward()
        self.critic_optimizer.step()

        # Log losses to wandb
        # wandb.log({
        #     "actor_loss": actor_loss.item(),
        #     "critic_loss": critic_loss.item()
        # })

        # Use Log_prob as loss instead of MSE
        # Actor loss is the negative of the predicted returns
        # Value loss is the "KL" loss between the predicted value and the actual value 
        return actor_loss, critic_loss # Return the world model loss, actor loss, critic loss

    def rollout(
        self,
    ):
        total_rewards = 0
        for t in range(self.batch_train_freq):
            self.num_timesteps += 1
            action = self.sample_action(torch.cat([self.prev_state.squeeze(), self.prev_latent_space.squeeze()], dim = -1).to(device))
            action = torch.tensor(action, dtype=torch.float32)
            if action.dim() == 1:
                action = action.reshape(1, action.shape[0])
            timestep = self.env.step(action.cpu())
            obs = torch.tensor(self.env.physics.render(camera_id=0, height=128, width=192).copy())
            obs = obs.reshape(1, obs.shape[0], obs.shape[1], obs.shape[2]).detach()
            action = action.reshape(1, action.shape[0], action.shape[1])
            states = self.RSSM(
                self.prev_state.to(device), 
                action.to(device), 
                self.prev_latent_space.to(device), 
                nonterminals=1-timestep.last(), 
                observations=obs.to(device)
            )

            # print(f"States {states}")
            if obs is not None:
                latent_spaces, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, decoded_observations, rewards = states
            else:
                latent_spaces, prior_states, prior_means, prior_std_devs, rewards = states

            self.prev_state = posterior_states
            self.prev_latent_space = latent_spaces

            self.replayBuffer.add(self.last_obs, action, timestep.reward, obs, timestep.last())
            self.last_obs = obs

            total_rewards += timestep.reward

        # Log total rewards after every rollout
        wandb.log({"total_rewards": total_rewards, "num_timesteps": self.num_timesteps})

    def train(
        self,
        timesteps : int,
        num_points : int,
        data_length : int,
        update_steps : int = 100,
    ):
        self.num_points = num_points
        self.data_length = data_length
        obs = self.env.reset()
        render = self.env.physics.render(camera_id=0, height=128, width=192)
        self.last_obs = torch.tensor(render.copy())
        self.prev_state = torch.zeros((1, self.RSSM.state_dim))
        self.prev_latent_space = torch.zeros((1, self.RSSM.latent_dim))

        self.num_timesteps = 0
        total_rewards = 0

        while(self.num_timesteps < self.steps_of_sampling):
            self.rollout()
            obs = self.env.reset()
            render = self.env.physics.render(camera_id=0, height=128, width=192)
            self.last_obs = torch.tensor(render.copy())

        obs = self.env.reset()
        render = self.env.physics.render(camera_id=0, height=128, width=192)
        self.last_obs = torch.tensor(render.copy())
        self.prev_state = torch.zeros((1, self.RSSM.state_dim))
        self.prev_latent_space = torch.zeros((1, self.RSSM.latent_dim))

        while (self.num_timesteps < timesteps):
            # wandb.init(project="dreamer_training", reinit=True)
            self.rollout()
            total_actor_loss = 0
            total_critic_loss = 0
            total_reward_loss = 0
            total_kl_loss = 0
            total_decoder_loss = 0
            for i in range(update_steps):
                beliefs, states, actions, reward_loss, kl_loss, decoder_loss = self.model_update()

                # The data that the agent update receives should be the encoded space already to save memory
                beliefs = beliefs.detach()
                states = states.detach()
                actions = actions.detach()
                actor_loss, critic_loss = self.agent_update(beliefs, states, actions)
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_reward_loss += reward_loss.item()
                total_kl_loss += kl_loss.item()
                total_decoder_loss += decoder_loss.item()
                # Log training progress to wandb
                wandb.log({
                    "num_timesteps": self.num_timesteps,
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "reward_loss" : reward_loss,
                    "observation_loss" : decoder_loss,
                    "kl_loss" : kl_loss.item()
                })

            avg_actor_loss = total_actor_loss / update_steps
            avg_critic_loss = total_critic_loss / update_steps
            avg_reward_loss = total_reward_loss / update_steps
            avg_kl_loss = total_kl_loss / update_steps
            avg_decoder_loss = total_decoder_loss / update_steps

            print(f"Timestep: {self.num_timesteps}, Avg Actor Loss: {avg_actor_loss}, Avg Critic Loss: {avg_critic_loss}, Avg Reward Loss: {avg_reward_loss}, Avg KL Loss: {avg_kl_loss}, Avg Decoder Loss: {avg_decoder_loss}")

            obs = self.env.reset()
            render = self.env.physics.render(camera_id=0, height=128, width=192)
            self.last_obs = torch.tensor(render.copy())
            self.prev_state = torch.zeros((1, self.RSSM.state_dim))
            self.prev_latent_space = torch.zeros((1, self.RSSM.latent_dim))


        return
    

    ### NEED TO EDIT THIS SO THAT REPRESENTATION MODEL ENCODES THE VALUES
    def sample_action(
        self,
        pixels : torch.Tensor,
        predict_mode : bool = False
    ) -> torch.Tensor:
        if (self.num_timesteps < self.steps_of_sampling):
            action_spec = self.env.action_spec()
            random_action = np.random.uniform(
                low=action_spec.minimum, 
                high=action_spec.maximum, 
                size=action_spec.shape
            )
            return random_action
        elif not predict_mode:
            predict = self.actor(pixels).detach()
            return predict + 0.3 * torch.randn_like(predict).detach()
        else:
            return self.actor(pixels).detach()


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
        # Next, we need to calculate the predicted targets of the next states (This is just current_reward + (1 - lambda) * gamma * next_value)        
        targets = pred_rewards + (1 - _lambda) * pred_values
        # Since we are using TD-lambda for finding the returns, this essentially correspond to the point that the returns on to 
        curr_val = last_reward
        outputs = []

        for i in range(pred_rewards.shape[1] - 1, -1, -1):
            curr_val = targets[i] + _lambda * curr_val
            outputs.append(curr_val)
        outputs = torch.stack(outputs, dim = 1)
        outputs = torch.flip(outputs, [0])
        # print(f"outputs: {outputs}")
        return outputs
        
    def save_models(self, num_timestep):
        self.actor.save_model(num_timestep)
        self.critic.save_model(num_timestep)

        with gzip.open(f"Buffers/buffer{num_timestep}", 'wb') as f:
            pickle.dump(self.replayBuffer, f)

    def load_model(self, num_timestep):
        self.actor.load_model(num_timestep)
        self.critic.load_model(num_timestep)

        with gzip.open(f"Buffers/buffer{num_timestep}", 'rb') as f:
            self.memory = pickle.load(f)

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
        x = nn.ELU()(self.l1(input))
        x = nn.ELU()(self.l2(x))
        if not self.action_model:  # For the value model
            mean, std = torch.chunk(self.l3(x), 2, dim=-1)
            
            # Ensure std is positive by applying softplus or another positive activation
            std = F.softplus(std) + 1e-6  # Add epsilon to avoid zero std
            
            # Construct a diagonal covariance matrix from std
            cov_mat = torch.diag_embed(std**2)
            
            return MultivariateNormal(mean, cov_mat)
        else: # For the actor model
            mean, std = torch.chunk(self.l3(x), 2, dim = -1)
            action = torch.tanh(mean + std.detach() * torch.randn_like(mean))
            return action

    def save_model(self, num_steps):
        if self.action_model:
            model_path = f"ModelCheckpoint/actor{num_steps}.pth"
        else:
            model_path = f"ModelCheckpoints/critic{num_steps}.pth"
        torch.save(self.state_dict(), model_path)

    def load_model(self, num_steps):
        if self.action_model:
            model_path = f"ModelCheckpoint/actor{num_steps}.pth"
        else:
            model_path = f"ModelCheckpoint/critic{num_steps}.pth"
        self.load_state_dict(torch.load(model_path))