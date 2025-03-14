import gymnasium as gym
from lunar_agent import LunarAgent
import numpy as np
from nn import QNetwork, ReplayBuffer
import torch

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

batch_size = 32
num_episodes = 1000
target_update_frequency = 10
min_epsilon = 0.01
epsilon_decay = 0.995
epsilon = 1.0
# Create a new instance of the QNetwork
loaded_q_network = QNetwork(state_dim, action_dim)

# Load the saved parameters into the new model
loaded_q_network.load_state_dict(torch.load('q_network.pth'))

# Set the model to evaluation mode
loaded_q_network.eval()

state, _ = env.reset()
state = np.array(state)
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = loaded_q_network(state_tensor)
        action = torch.argmax(q_values).item()

    next_state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    state = np.array(next_state)
    
    env.render()
    
    if done or truncated:
        break

print(f"Total Reward: {total_reward}")
env.close()
