from collections import defaultdict
import gymnasium as gym
from nn import QNetwork, ReplayBuffer
import random
import torch
import torch.optim as optim
import torch.nn as nn

# target_q = reward + γ × (max future Q-value) × (1−done)

class LunarAgent:
    def __init__(self, env, action_size, state_size, epsilon, batch_size, gama, min_epsilon, epsilon_decay):
        self.env = env
        self.q_network = QNetwork(state_size, action_size)
        self.memory = ReplayBuffer(100000)
        self.target_network = QNetwork(state_size, action_size)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gama = gama 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay


    def get_action(self, state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            return action

    def update(self, episode, target_update_frequency):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Compute current Q values
            q_values = self.q_network(states)
            # To extract the Q-value corresponding to the specific action taken in each state.
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute next Q values using the target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                target_q = rewards + self.gama * max_next_q * (1 - dones)

            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

    def push_buffer(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self):
        torch.save(self.q_network.state_dict(), 'q_network.pth')