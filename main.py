# Using the dm_control suite instead of using gym!!!
import yaml
from dm_control import suite
from dm_control import viewer
from Dreamer import Dreamer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == '__main__':
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