import gymnasium as gym
from lunar_agent import LunarAgent
import numpy as np

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

batch_size = 32
num_episodes = 500
target_update_frequency = 10
min_epsilon = 0.01
epsilon_decay = 0.995
epsilon = 1.0

agent = LunarAgent(env=env, action_size=action_dim, state_size=state_dim, epsilon=epsilon, batch_size=batch_size, gama=0.99, 
                   min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.array(state)
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_state)
        total_reward += reward

        agent.push_buffer(state, action, reward, next_state, done)
        state = next_state

        agent.update(episode, target_update_frequency)

        if done or truncated:
            break
    
    agent.decay_epsilon()
    
    print(f"Episode: {episode}, Total Reward: {total_reward:.2f}")

agent.save_model()

env.close()