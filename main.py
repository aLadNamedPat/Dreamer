# Using the dm_control suite instead of using gym!!!
from dm_control import suite
from dm_control import viewer
from Dreamer import Dreamer

if __name__  == '__main__':
    env = suite.load(domain_name="walker", task_name="run")
    dreamer = Dreamer(env, state_dims = 50, latent_dims = 50, observation_dim=3, 
                      o_feature_dim = 50, reward_dim = 1)

    dreamer.train()