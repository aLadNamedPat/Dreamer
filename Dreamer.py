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
from value_functions import compute_Vlambda
import cv2

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')


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
            img_h : int,
            img_w : int,
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
        self.img_h = img_h
        self.img_w = img_w
        self.device = device

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
            o_dim = (self.img_h, self.img_w),
            latent_dim=self.latent_dims,
            reward_dim=self.reward_dim
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr =8e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)
        self.RSSM_optimizer = torch.optim.Adam(self.RSSM.parameters(), lr =6e-4)

    # Sparkly fun things going on here
    # def latent_imagine(self, latents, posterior, horizon : int):
    # # Latent imagination receives the latents and the posterior where the latents are the probability distribution over possible events whereas the posterior is the deterministic

    # # Posterior is a M x N vector representing the state at each different index
    # # Latent is a M x N vector representing the latent at each different index
    #     x, y = posterior.shape

    #     # imagined_state = posterior.reshape(x * y, -1)
    #     # imagined_latent = latents.reshape(x * y, -1)
    #     imagined_state = posterior
    #     imagined_latent = latents
    #     action = self.actor(torch.cat([imagined_state, imagined_latent], -1))
    #     # print(f"Action Reshape {action.reshape(x, y, -1)}")

    #     latent_list = [imagined_latent]
    #     state_list = [imagined_state]
    #     action_list = [action]

    #     for _ in range(horizon):
    #         state = self.RSSM(imagined_state, action_list, imagined_latent)
    #         imagined_state, imagined_latent = state[0], state[1]
    #         action = self.actor(torch.cat([imagined_state, imagined_latent], -1))
    #         # action.reshape(x, y, -1)
    #         latent_list.append(imagined_latent)
    #         state_list.append(imagined_state)
    #         action_list.append(action)
        
    #     latent_list = torch.stack(latent_list, dim = 0)
    #     state_list = torch.stack(state_list, dim = 0)
    #     action_list = torch.stack(action_list, dim = 0)

    #     return latent_list, state_list, action_list

    # Will return new trajectories of states and actions that will be used to train our model
    def model_update(self):

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards_real, next_states, dones = self.replayBuffer.sample(self.batch_size, self.sample_steps)
        # print(states.shape)
        # print(actions.shape)
        # print(rewards_real.shape)
        # print(dones.shape)

        # Get the initial state and latent space

        prev_state = torch.zeros((self.batch_size, self.RSSM.state_dim)).to(self.device)
        prev_latent_space = torch.zeros((self.batch_size, self.RSSM.latent_dim)).to(self.device)
        # Forward pass through the RSSM
        # print(f"Dones: {dones}")
        # print(f"actions: {actions.squeeze()}")
        # print(f"states: {prev_state.shape}")

        # import pdb; pdb.set_trace()

        # print(f"STATES : {states}")
        latent_spaces, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, decoded_observations, rewards = self.RSSM(
            prev_state.to(device),
            actions.squeeze().float().to(device),
            prev_latent_space.to(device), 
            nonterminals=torch.logical_not(dones).to(device), 
            observations=states.to(device)
        )


        # Calculate the MSE loss for observation and decoded observation
        mse_loss = nn.MSELoss()
        # print(f"States : {states.shape}")
        # print(f"Decoded  : {decoded_observations.shape}")
        # print(f"States shape: {states.shape}")
        # print("State: ", states[0])
        # print("Decoded: ", decoded_observations[0])
        # import matplotlib.pyplot as plt

        # # Select the first state and decoded observation in the batch to plot
        # state_to_plot = states[0, 0].cpu().detach().numpy()
        # decoded_to_plot = decoded_observations[0, 0].cpu().detach().numpy()

        # # Create a figure with two subplots
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # # Plot the state using matplotlib
        # axes[0].imshow(state_to_plot)
        # axes[0].set_title("State at index [0, 0]")
        # axes[0].axis('off')

        # # Plot the decoded observation using matplotlib
        # axes[1].imshow(decoded_to_plot)
        # axes[1].set_title("Decoded at index [0, 0]")
        # axes[1].axis('off')

        # # Show the plots
        # plt.show()
        observation_loss = mse_loss(states.float(), decoded_observations)
        
        # Calculate the KL divergence loss between the prior and posterior distributions
        kl_loss = torch.distributions.kl_divergence(
            torch.distributions.Normal(posterior_means, posterior_std_devs),
            torch.distributions.Normal(prior_means, prior_std_devs)
        ).mean()

        # print(f"Rewards Real Shape: {rewards_real.shape}")
        # print(f"Rewards Shape: {rewards.shape}")
        # Calculate the reward loss between the real rewards and the imagined rewards
        
        reward_loss = mse_loss(rewards_real.float().unsqueeze(2), rewards.float())
        # print(f"Reward Loss: {reward_loss.item()}")

        total_loss = observation_loss + 0.1 *  kl_loss + reward_loss

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
        return latent_spaces, prior_states, posterior_states, actions, reward_loss, kl_loss, observation_loss, rewards
    
    def agent_update(
            self,
            beliefs,
            states,
            imagined_rewards
        ):
        
        # print(f"Started Update")
        # print(f"States shape: {states.shape}")
        # print(f"Beliefs shape: {beliefs.shape}")
        imagined_beliefs, imagined_states, imagined_actions, imagined_rewards = self.imagine_rollout(
            start_belief=beliefs[:, 0],
            start_state=states[:, 0],
            horizon=self.horizon
        )

        # print(f"Imagined States shape: {imagined_states.shape}")
        # print(f"Imagined Beliefs shape: {imagined_beliefs.shape}")

        # Compute critic rewards for imagined states and beliefs
        imagined_beliefs = imagined_beliefs.transpose(0, 1)
        imagined_states = imagined_states.transpose(0, 1)
        imagined_actions = imagined_actions.transpose(0, 1)
        imagined_rewards = imagined_rewards.transpose(0, 1)
        imagined_rewards = imagined_rewards.unsqueeze(2)
        # print(f"Imagined Rewards shape: {imagined_rewards.shape}")
        critic_rewards, distribution = self.critic(torch.cat([imagined_states.detach(), imagined_beliefs.detach()], dim=-1))
        # print(f"Critic rewards shape: {critic_rewards.shape}")

        # Ensure critic_rewards requires grad
        

        imagined_values = compute_Vlambda(
            states=imagined_states,
            rewards=imagined_rewards,
            tau=0,
            H=self.horizon,
            gamma=self.gamma,
            lam=self.lambda_,
            value_fn_rewards=critic_rewards
        )

        imagined_values = imagined_values.requires_grad_()
        critic_rewards = critic_rewards.requires_grad_()
        wandb.log({"critic_rewards_sum": critic_rewards.sum().item(), "num_timesteps": self.num_timesteps})
        # print(f"Imagined values: {imagined_values.shape}")

        # Update actor parameters (ϕ)
        # print(f"Imagined values shape: {imagined_values.shape}")
        
        actor_loss = -imagined_values.sum(dim=1).mean(dim=0)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  
        self.actor_optimizer.step()

        # Update critic parameters (ψ)
        target_values = imagined_values.detach()
        # print(f"Target values shape: {target_values.shape}")
        critic_loss = 0.5 * (critic_rewards - target_values).pow(2).sum(dim=1).mean(dim=0)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss, critic_loss
    

    def imagine_rollout(self, start_belief, start_state, horizon):
        beliefs = []
        states = []
        actions = []
        rewards = []
        b, s = start_belief, start_state
        for t in range(horizon):
            a = self.sample_action(torch.cat([b, s], dim=-1).to(self.device))
            if a.ndimension() == 2:
                a = a.unsqueeze(1)
            a = torch.tensor(a, dtype=torch.float32).to(self.device)
            
            # forward the RSSM with (b, s, a) => next_b, next_s, 
            b = b.squeeze(1)
            s = s.squeeze(1)
            rssm_outputs = self.RSSM(b, a, s)
            b, s = rssm_outputs[:2]
            r = rssm_outputs[-1]
            # store them
            beliefs.append(b)
            states.append(s)
            actions.append(a)
            rewards.append(r)

        beliefs = torch.stack(beliefs).squeeze()
        states = torch.stack(states).squeeze()
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).squeeze()

        # print("Beliefs shape:", beliefs.shape)
        # print("States shape:", states.shape)
        # print("Actions shape:", actions.shape)
        # print("Rewards shape:", rewards.shape)
        
        return beliefs, states, actions, rewards


    def rollout(
        self,
        use_RSSM = False
    ):
        total_rewards = 0
        self.prev_state = torch.zeros((1, self.RSSM.state_dim)).to(self.device)
        self.prev_latent_space = torch.zeros((1, self.RSSM.latent_dim)).to(self.device)
        for t in range(self.sample_steps):
            self.num_timesteps += 1
            action = self.sample_action(torch.cat([self.prev_state.squeeze(), self.prev_latent_space.squeeze()], dim=-1).to(self.device))
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            if action.dim() == 1:
                action = action.reshape(1, action.shape[0])
            timestep = self.env.step(action.cpu())
            # print("TIMESTEP: ", timestep)
            obs = torch.tensor(self.env.physics.render(camera_id=0, height=self.img_h, width=self.img_w).copy()).to(self.device)
            obs = obs.reshape(1, obs.shape[0], obs.shape[1], obs.shape[2]).detach()
            action = action.reshape(1, action.shape[0], action.shape[1])
            if use_RSSM:
                states = self.RSSM(
                    self.prev_state.to(self.device).unsqueeze(0), 
                    action.to(self.device),
                    self.prev_latent_space.to(self.device).unsqueeze(0), 
                    nonterminals=(1-timestep.last()), 
                    observations=obs.to(self.device).unsqueeze(0),
                )

                if obs is not None:
                    
                    latent_spaces, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, decoded_observations, rewards = states
                    self.prev_state = posterior_states[:, -1]
                    self.prev_latent_space = latent_spaces[:, -1]
                    # import matplotlib.pyplot as plt

                    
                    
                    # print("Decoded Observation:", decoded_observations)
                    # print("Observation shape:", obs.shape)
                    # print("Decoded Observation shape:", decoded_observations.shape)
                    # print("Observation min:", obs.min().item(), "max:", obs.max().item())
                    # print("Decoded Observation min:", decoded_observations.min().item(), "max:", decoded_observations.max().item())
                    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    # axes[0].imshow(obs[0].cpu().detach().numpy())
                    # axes[0].axis('off')
                    # axes[0].set_title("Current Observation")
                    
                    # axes[1].imshow(decoded_observations[0, -1].cpu().detach().numpy())
                    # axes[1].axis('off')
                    # axes[1].set_title("Decoded Observation")

                    # plt.show()
                else:
                    latent_spaces, prior_states, prior_means, prior_std_devs, rewards = states
                    self.prev_state = prior_states[:, -1]
                    self.prev_latent_space = latent_spaces[:, -1]

            self.replayBuffer.add(self.last_obs, action, timestep.reward, obs, timestep.last())
            
            self.last_obs = obs

            total_rewards += timestep.reward

        if (self.num_timesteps >= self.steps_of_sampling):
            wandb.log({"total_rewards": total_rewards, "num_timesteps": self.num_timesteps})

    def train(
        self,
        timesteps : int,
        num_points : int,
        data_length : int,
        update_steps : int = 15,
        video_interval : int = 100,  # New parameter for video saving interval
        video_path : str = "training_video.mp4"  # New parameter for video path
    ):
        self.num_points = num_points
        self.data_length = data_length
        obs = self.env.reset()
        render = self.env.physics.render(camera_id=0, height=self.img_h, width=self.img_w)
        self.last_obs = torch.tensor(render.copy()).to(self.device)
        self.prev_state = torch.zeros((1, self.RSSM.state_dim)).to(self.device)
        self.prev_latent_space = torch.zeros((1, self.RSSM.latent_dim)).to(self.device)

        self.num_timesteps = 0
        total_rewards = 0

        while(self.num_timesteps < self.steps_of_sampling):
            self.rollout()
            obs = self.env.reset()
            render = self.env.physics.render(camera_id=0, height=self.img_h, width=self.img_w)
            self.last_obs = torch.tensor(render.copy()).to(self.device)

        obs = self.env.reset()
        render = self.env.physics.render(camera_id=0, height=self.img_h, width=self.img_w)
        self.last_obs = torch.tensor(render.copy()).to(self.device)

        while (self.num_timesteps < timesteps):
            total_actor_loss = 0
            total_critic_loss = 0
            total_reward_loss = 0
            total_kl_loss = 0
            total_decoder_loss = 0
            for i in range(update_steps):
                print(f"i : {i}")

                ###*Dynamics Learning*###
                beliefs, states, posterior_states, actions, reward_loss, kl_loss, decoder_loss, imagined_rewards = self.model_update()

                ###*Behavior Learning*###
                actor_loss, critic_loss = self.agent_update(beliefs, states, imagined_rewards)

                total_reward_loss += reward_loss.item()
                total_kl_loss += kl_loss.item()
                total_decoder_loss += decoder_loss.item()
                print(f"Reward Loss: {reward_loss}")
                wandb.log({
                    "num_timesteps": self.num_timesteps,
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "reward_loss" : reward_loss.item(),
                    "observation_loss" : decoder_loss,
                    "kl_loss" : kl_loss.item()
                })


                # Save observations to video at specified intervals
                if self.num_timesteps % video_interval == 0:
                    self.save_observations_to_video(eval_steps=self.sample_steps, video_path=video_path)

            self.rollout(use_RSSM=True)

            avg_actor_loss = total_actor_loss / update_steps
            avg_critic_loss = total_critic_loss / update_steps
            avg_reward_loss = total_reward_loss / update_steps
            avg_kl_loss = total_kl_loss / update_steps
            avg_decoder_loss = total_decoder_loss / update_steps

            print(f"Timestep: {self.num_timesteps}, Avg Actor Loss: {avg_actor_loss}, Avg Critic Loss: {avg_critic_loss}, Avg Reward Loss: {avg_reward_loss}, Avg KL Loss: {avg_kl_loss}, Avg Decoder Loss: {avg_decoder_loss}")

            obs = self.env.reset()
            render = self.env.physics.render(camera_id=0, height=self.img_h, width=self.img_w)
            self.last_obs = torch.tensor(render.copy()).to(self.device)
            self.prev_state = torch.zeros((1, self.RSSM.state_dim)).to(self.device)
            self.prev_latent_space = torch.zeros((1, self.RSSM.latent_dim)).to(self.device)

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
            return (predict + 0.3 * torch.randn_like(predict).detach())
        else:
            return self.actor(pixels).detach()

    def save_observations_to_video(self, eval_steps: int, video_path: str):
        # Append num_timesteps to the video_path
        video_path_with_timesteps = f"{video_path}_{self.num_timesteps}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path_with_timesteps, fourcc, 20.0, (self.img_w, self.img_h))

        obs = self.env.reset()
        render = self.env.physics.render(camera_id=0, height=self.img_h, width=self.img_w)
        self.last_obs = torch.tensor(render.copy()).to(self.device)
        self.prev_state = torch.zeros((1, self.RSSM.state_dim)).to(self.device)
        self.prev_latent_space = torch.zeros((1, self.RSSM.latent_dim)).to(self.device)

        for t in range(eval_steps):
            action = self.sample_action(torch.cat([self.prev_state.squeeze(), self.prev_latent_space.squeeze()], dim=-1).to(self.device), predict_mode=True)
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            if action.dim() == 1:
                action = action.reshape(1, action.shape[0])
            # Ensure action has the correct dimensions
            if action.dim() == 2:
                action = action.unsqueeze(1)  # Add a dimension if necessary
            timestep = self.env.step(action.cpu())
            obs = torch.tensor(self.env.physics.render(camera_id=0, height=self.img_h, width=self.img_w).copy()).to(self.device)
            obs = obs.reshape(1, obs.shape[0], obs.shape[1], obs.shape[2]).detach()
            self.prev_state, self.prev_latent_space = self.RSSM(self.prev_state, action, self.prev_latent_space)[:2]

            frame = obs[0].cpu().numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

            if timestep.last():
                break

        out.release()

# Help from https://github.com/juliusfrost/dreamer-pytorch/blob/main/dreamer/algos/dreamer_algo.py for finding returns
# http://www.incompleteideas.net/book/RLbook2020.pdf

    # def find_predicted_returns(
    #     self,
    #     pred_rewards,
    #     pred_values,
    #     last_value,
    #     _lambda,
    #     gamma
    # ):

    #     returns = []

    #     curr_val = last_value
        
    #     print(f"Predicted rewards shape: {pred_rewards.shape}")
    #     print(f"Predicted values shape: {pred_values.shape}")
    #     for i in range(pred_rewards.shape[1] - 1, -1, -1):
    #         curr_val = pred_rewards[:, i] + gamma * ((1 - _lambda) * pred_values[:, i] + _lambda * curr_val)
    #         returns.append(curr_val)

    #     returns.reverse()

    #     return torch.stack(returns, dim=1)

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
                 input_dims: int, 
                 output_dims: int, 
                 mid_dims: int = 300, 
                 action_model: bool = False):
        super(DenseConnections, self).__init__()
        self.l1 = nn.Linear(input_dims, mid_dims)
        self.l2 = nn.Linear(mid_dims, mid_dims)
        self.l3 = nn.Linear(mid_dims, 2 * output_dims)

        self.action_model = action_model

    def forward(self, input: torch.Tensor):
        x = nn.ELU()(self.l1(input))
        x = nn.ELU()(self.l2(x))
        if not self.action_model:  # For the value model
            mean, std = torch.chunk(self.l3(x), 2, dim=-1)
            
            # Ensure std is positive by applying softplus or another positive activation
            std = F.softplus(std) + 1e-6  # Add epsilon to avoid zero std
            
            # Construct a diagonal covariance matrix from std
            cov_mat = torch.diag_embed(std**2)
            
            dist = MultivariateNormal(mean, cov_mat)
            sample = dist.rsample()  
            
            return sample, dist
        else:  # For the actor model
            mean, std = torch.chunk(self.l3(x), 2, dim=-1)
            mean = 5 * torch.tanh(mean)  # Scale the tanh mean by a factor of 5
            std = F.softplus(std) + 1e-6  # Ensure std is positive
            dist = MultivariateNormal(mean, torch.diag_embed(std**2))
            action = torch.tanh(dist.rsample())  # Transform using tanh
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