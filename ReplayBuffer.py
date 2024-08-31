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
            batch_states = torch.zeros((data_points))
            batch_actions = torch.zeros((data_points))
            batch_rewards = torch.zeros((data_points))
            batch_next_states = torch.zeros((data_points))
            batch_dones = torch.zeros((data_points))

            for i, idx in enumerate(sampled_indices):
                idx_sequence_states = torch.stack([exp[0] for exp in self.buffer[idx: idx + data_length]])
                idx_sequence_actions = torch.stack([exp[1] for exp in self.buffer[idx: idx + data_length]])
                idx_sequence_rewards = torch.stack([exp[2] for exp in self.buffer[idx: idx + data_length]])
                idx_sequence_next_states = torch.stack([exp[3] for exp in self.buffer[idx: idx + data_length]])
                idx_sequence_dones = torch.stack([exp[4] for exp in self.buffer[idx: idx + data_length]])

                batch_states[i] = idx_sequence_states
                batch_actions[i] = idx_sequence_actions
                batch_rewards[i] = idx_sequence_rewards
                batch_next_states[i] = idx_sequence_next_states
                batch_dones[i] = idx_sequence_dones
                
            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def get_size(self):
        return len(self.buffer)

