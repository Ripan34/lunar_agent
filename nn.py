from torch import nn
import numpy as np
import random
from collections import deque
import torch

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.f1 = nn.Linear(state_size, 128)
        self.f2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.f1(state))
        x = torch.relu(self.f2(x))

        return self.out(x)

 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)