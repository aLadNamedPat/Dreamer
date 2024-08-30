import torch
from collections import deque
import random

class Buffer():
    def __init__(self, buffer_size : int):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, rewards, next_state, done) -> None:
        self.buffer.append((state, action, rewards, next_state, done))

    # Supports both random sampling and random sampling with fixed data length
    def sample(self, data_points : int, data_length : int, random_flag : bool = False):
        if random_flag:
            sampled_experiences = random.sample(self.buffer, data_points)
            batch_states = torch.stack([exp[0] for exp in sampled_experiences])
            batch_actions = torch.stack([exp[1] for exp in sampled_experiences])
            batch_rewards = torch.stack([exp[2] for exp in sampled_experiences])
            batch_next_states = torch.stack([exp[3] for exp in sampled_experiences])
            batch_dones = torch.stack([exp[4] for exp in sampled_experiences])
            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
        else:
            sampled_indices = random.sample(range(len(self.buffer) - data_length + 1), data_points)
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []

            for idx in sampled_indices:
                for i in range(data_length):
                    exp = self.buffer[idx + i]
                    batch_states.append(exp[0])
                    batch_actions.append(exp[1])
                    batch_rewards.append(exp[2])
                    batch_next_states.append(exp[3])
                    batch_dones.append(exp[4])

            batch_states = torch.stack(batch_states)
            batch_actions = torch.stack(batch_actions)
            batch_rewards = torch.stack(batch_rewards)
            batch_next_states = torch.stack(batch_next_states)
            batch_dones = torch.stack(batch_dones)

            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def get_size(self):
        return len(self.buffer)

