
import gymnasium as gym
import numpy as np
import random
import time

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Q-table initialization
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Hyperparameters
episodes = 5000
max_steps = 100
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        # Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update rule
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state

        if done:
            break

    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Training completed!")
print("Final Q-Table:")
print(q_table)

# Evaluation
total_rewards = 0
eval_episodes = 100

for episode in range(eval_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = new_state
        total_rewards += reward

print(f"Average reward over {eval_episodes} episodes: {total_rewards/eval_episodes}")
