import torch
from collections import deque
import random

class Buffer():
    def __init__(
        self,
        buffer_size : int,
    ):
        self.buffer = deque(maxlen=buffer_size)


    def add(
        self,
        state,
        action,
        rewards,
        next_state,
        done) -> None:

        self.buffer.append((
            state,
            action,
            rewards,
            next_state,
            done))

    def sample(
        self,
        batch_size : int
    ):
        sampled_experiences = random.sample(self.buffer, batch_size)
        batch_states = torch.stack([exp[0] for exp in sampled_experiences])
        batch_actions = torch.stack([exp[1] for exp in sampled_experiences])
        batch_rewards = torch.stack([exp[2] for exp in sampled_experiences])
        batch_next_states = torch.stack([exp[3] for exp in sampled_experiences])
        batch_dones = torch.stack([exp[4] for exp in sampled_experiences])
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones


    def get_size(
        self
    ):
        return len(self.buffer)

