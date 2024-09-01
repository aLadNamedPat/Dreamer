# Using the dm_control suite instead of using gym!!!
import yaml
from dm_control import suite
# from dm_control import viewer
import os
os.environ['MUJOCO_GL'] = 'egl'
from Dreamer import Dreamer
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run(configurations, dreamer : Dreamer):
    env = suite.load(domain_name=configurations['env']['domain_name'], task_name=configurations['env']['task_name'])
    camera_id = 0
    timesteps = configurations['training']['timesteps']
    obs = env.reset()
    render = env.physics.render(camera_id=camera_id, height=120, width=160)
    last_obs = torch.tensor(render.copy())
    prev_state = torch.zeros((config['dreamer']['state_dims']))
    prev_latent_space = torch.zeros((config['dreamer']['latent_dims']))

    for _ in range(timesteps):
        action = dreamer.actor(torch.cat([prev_state, prev_latent_space]))
        timestep = env.step(action)
        obs = torch.tensor(env.physics.render(camera_id=0, height=120, width=160).copy())
    
        # Take a step in the environment
        time_step = env.step(action)
        done = False
        latent_spaces, prior_states, prior_means, prior_std_devs, \
        posterior_states, posterior_means, posterior_std_devs, \
        decoded_observations, rewards = dreamer.RSSM(
            prev_state, 
            action, 
            prev_latent_space, 
            nonterminals=1-done, 
            observations=obs
        )

        prev_state = posterior_states
        prev_latent_space = latent_spaces

        print(time_step)
        # Render and display the current pixel observation

        plt.imshow(obs)
        plt.axis('off')
        plt.pause(0.01)  # Small pause to allow image display
 
        
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    config = load_config('config.yaml')

    env = suite.load(domain_name=config['env']['domain_name'], task_name=config['env']['task_name'])
    dreamer = Dreamer(
        env=env,
        state_dims=config['dreamer']['state_dims'],
        latent_dims=config['dreamer']['latent_dims'],
        observation_dim=tuple(config['dreamer']['observation_dim']),
        o_feature_dim=config['dreamer']['o_feature_dim'],
        reward_dim=config['dreamer']['reward_dim'],
        gamma=config['dreamer']['gamma'],
        lambda_=config['dreamer']['lambda_'],
        batch_size=config['dreamer']['batch_size'],
        batch_train_freq=config['dreamer']['batch_train_freq'],
        buffer_size=config['dreamer']['buffer_size'],
        sample_steps=config['dreamer']['sample_steps']
    )

    timesteps = config['training']['timesteps']
    num_points = config['training']['num_points']
    data_length = config['training']['data_length']

    dreamer.train(timesteps, num_points, data_length)

    run(config, dreamer)