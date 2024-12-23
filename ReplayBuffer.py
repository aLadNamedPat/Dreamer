import torch
from collections import deque
import random

class Buffer():
    def __init__(self, buffer_size : int):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.curr_idx = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, rewards, next_state, done) -> None:
        self.buffer.append((
            torch.tensor(state.squeeze()).to(self.device),
            torch.tensor(action).to(self.device),
            torch.tensor(rewards).to(self.device),
            torch.tensor(next_state).to(self.device),
            torch.tensor(done).to(self.device)
        ))
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
    def sample(self, batch_size : int, data_length : int, random_flag : bool = False):
        # if random_flag:
        #     sampled_experiences = random.sample(self.buffer, batch_size)
        #     batch_states = torch.stack([exp[0].squeeze(0) if exp[0].dim() == 4 else exp[0] for exp in sampled_experiences])
        #     batch_actions = torch.stack([exp[1] for exp in sampled_experiences])
        #     batch_rewards = torch.stack([exp[2] for exp in sampled_experiences])
        #     batch_next_states = torch.stack([exp[3].squeeze(0) if exp[3].dim() == 4 else exp[3] for exp in sampled_experiences])
        #     batch_dones = torch.stack([exp[4] for exp in sampled_experiences])
            
        #     return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
        # else:

        sampled_indices = [self.sample_idx(data_length) for _ in range(batch_size)]
        # print(f"Batch Size : {batch_size}")
        # print(f"Data Length : {data_length}")
        # print(f"Buffer : {self.buffer}")
        batch_states = torch.zeros((batch_size, data_length) + self.buffer[0][0].shape).to(self.device)
        batch_actions = torch.zeros((batch_size, data_length) + self.buffer[0][1].shape).to(self.device)
        batch_rewards = torch.zeros((batch_size, data_length) + self.buffer[0][2].shape).to(self.device)
        batch_next_states = torch.zeros((batch_size, data_length) + self.buffer[0][3].shape).to(self.device)
        batch_dones = torch.zeros((batch_size, data_length) + self.buffer[0][4].shape).to(self.device)
        # print(f"Batch States Shape: {batch_states.shape}")
        # print(f"Batch Actions Shape: {batch_actions.shape}")
        # print(f"Batch Rewards Shape: {batch_rewards.shape}")
        # print(f"Batch Next States Shape: {batch_next_states.shape}")
        # print(f"Batch Dones Shape: {batch_dones.shape}")

        # batch_states = torch.zeros((batch_size, data_length))
        # batch_actions = torch.zeros((batch_size, data_length))
        # batch_rewards = torch.zeros((batch_size, data_length))
        # batch_next_states = torch.zeros((batch_size, data_length))
        # batch_dones = torch.zeros((batch_size, data_length))

        # print(batch_states.shape)
        # print(batch_actions.shape)
        # print(batch_rewards.shape)
        # print(batch_next_states.shape)
        # print(batch_dones.shape)
        # print(self.curr_idx)
        # print(sampled_indices)
        
        # print(f"Sampled Indices: {sampled_indices}")
        for i, idxs in enumerate(sampled_indices):
            # print(f"Sampling index: {i}")
            idx_sequence_states = torch.stack([self.buffer[idx][0] / 255.0 for idx in idxs])
            idx_sequence_actions = torch.stack([self.buffer[idx][1] for idx in idxs])
            idx_sequence_rewards = torch.stack([self.buffer[idx][2] for idx in idxs])
            idx_sequence_next_states = torch.stack([self.buffer[idx][3] for idx in idxs])
            idx_sequence_dones = torch.stack([self.buffer[idx][4] for idx in idxs])
            
            # print(f"States shape: {idx_sequence_states.shape}")
            # print(f"Actions shape: {idx_sequence_actions.shape}")
            # print(f"Rewards shape: {idx_sequence_rewards.shape}")
            # print(f"Next States shape: {idx_sequence_next_states.shape}")
            # print(f"Dones shape: {idx_sequence_dones.shape}")
            # print(f"idx_sequence_states: {idx_sequence_states}")
            # print(f"Rewards : {idx_sequence_rewards}")
            batch_states[i] = idx_sequence_states
            
            batch_actions[i] = idx_sequence_actions
            batch_rewards[i] = idx_sequence_rewards
            batch_next_states[i] = idx_sequence_next_states
            batch_dones[i] = idx_sequence_dones

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def get_size(self):
        return len(self.buffer)
