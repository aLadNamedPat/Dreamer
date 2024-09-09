import torch
from collections import deque
import random

class Buffer():
    def __init__(self, buffer_size : int):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.curr_idx = 0

    def add(self, state, action, rewards, next_state, done) -> None:
        self.buffer.append((torch.tensor(state.squeeze()), torch.tensor(action), torch.tensor(rewards), torch.tensor(next_state), torch.tensor(done)))
        self.curr_idx = (self.curr_idx + 1) % self.buffer_size


    def sample_idx(self, data_length : int):
        val_idx = False
        while not val_idx:
            sampled_ind = torch.randint(0, len(self.buffer) - data_length + 1, (1,))
            sampled_ind = sampled_ind.item()
            indices = torch.arange(sampled_ind, sampled_ind + data_length) % self.buffer_size
            val_idx = not self.curr_idx in indices[1:]
        return indices
    
    # Supports both random sampling and random sampling with fixed data length
    def sample(self, data_points : int, data_length : int, random_flag : bool = False):
        # if random_flag:
        #     sampled_experiences = random.sample(self.buffer, data_points)
        #     batch_states = torch.stack([exp[0].squeeze(0) if exp[0].dim() == 4 else exp[0] for exp in sampled_experiences])
        #     batch_actions = torch.stack([exp[1] for exp in sampled_experiences])
        #     batch_rewards = torch.stack([exp[2] for exp in sampled_experiences])
        #     batch_next_states = torch.stack([exp[3].squeeze(0) if exp[3].dim() == 4 else exp[3] for exp in sampled_experiences])
        #     batch_dones = torch.stack([exp[4] for exp in sampled_experiences])
            
        #     return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
        # else:

        sampled_indices = [self.sample_idx(data_length) for _ in range(data_points)]

        batch_states = torch.zeros((data_points, data_length) + self.buffer[0][0].shape)
        batch_actions = torch.zeros((data_points,data_length) + self.buffer[0][1].shape)
        batch_rewards = torch.zeros((data_points, data_length) + self.buffer[0][2].shape)
        batch_next_states = torch.zeros((data_points, data_length) + self.buffer[0][3].shape)
        batch_dones = torch.zeros((data_points, data_length) + self.buffer[0][4].shape)

        # batch_states = torch.zeros((data_points, data_length))
        # batch_actions = torch.zeros((data_points, data_length))
        # batch_rewards = torch.zeros((data_points, data_length))
        # batch_next_states = torch.zeros((data_points, data_length))
        # batch_dones = torch.zeros((data_points, data_length))

        print(batch_states.shape)
        print(batch_actions.shape)
        print(batch_rewards.shape)
        print(batch_next_states.shape)
        print(batch_dones.shape)
        print(self.curr_idx)
        print(sampled_indices)
        for i, idxs in enumerate(sampled_indices):
            idx_sequence_states = torch.stack([self.buffer[idx][0] for idx in idxs])
            idx_sequence_actions = torch.stack([self.buffer[idx][1] for idx in idxs])
            idx_sequence_rewards = torch.stack([self.buffer[idx][2] for idx in idxs])
            idx_sequence_next_states = torch.stack([self.buffer[idx][3] for idx in idxs])
            idx_sequence_dones = torch.stack([self.buffer[idx][4] for idx in idxs])

            batch_states[i] = idx_sequence_states
            batch_actions[i] = idx_sequence_actions
            batch_rewards[i] = idx_sequence_rewards
            batch_next_states[i] = idx_sequence_next_states
            batch_dones[i] = idx_sequence_dones

            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def get_size(self):
        return len(self.buffer)

