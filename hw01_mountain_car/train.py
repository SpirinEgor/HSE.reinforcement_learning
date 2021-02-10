import random

import numpy as np
from gym import make

SEED = 7
GRID_SIZE_X = 30
GRID_SIZE_Y = 30

TRANSITIONS = 4_000_000
START_EPS = 0.3
REWARD_GAMMA = 0.98

GAMMA = 0.98
ALPHA = 0.1


# Simple discretization 
def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X - 1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y - 1)
    return x + GRID_SIZE_X * y


class QLearning:
    def __init__(self, state_dim, action_dim, gamma: float = 0.98, alpha: float = 0.1):
        self.qlearning_estimate = np.zeros((state_dim, action_dim)) + 2.
        self._gamma = gamma
        self._alpha = alpha

    def update(self, transition):
        state, action, next_state, reward, done = transition
        bellman = reward + self._gamma * self.qlearning_estimate[next_state].max()
        self.qlearning_estimate[state][action] *= (1 - self._alpha)
        self.qlearning_estimate[state][action] += self._alpha * bellman

    def act(self, state):
        return self.qlearning_estimate[state].argmax()

    def save(self, path):
        np.save(path, self.qlearning_estimate)


def evaluate_policy(agent, episodes=5, render: bool = False):
    env = make("MountainCar-v0")
    env.seed(SEED)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(transform_state(state)))
            if render:
                env.render()
            total_reward += reward
        returns.append(total_reward)
    return returns


def main():
    env = make("MountainCar-v0")
    env.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    ql = QLearning(state_dim=GRID_SIZE_X * GRID_SIZE_Y, action_dim=3, gamma=GAMMA, alpha=ALPHA)

    state = env.reset()
    table_state = transform_state(state)

    trajectory = []
    best_score = -1e9
    eps = np.linspace(START_EPS, 0, TRANSITIONS)

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps[i]:
            action = env.action_space.sample()
        else:
            action = ql.act(table_state)

        next_state, reward, done, _ = env.step(action)
        reward += 1000 * (REWARD_GAMMA * abs(next_state[1]) - abs(state[1]))
        table_next_state = transform_state(next_state)

        trajectory.append((table_state, action, table_next_state, reward, done))

        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []

        state = next_state if not done else env.reset()
        table_state = transform_state(state)

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(ql, 5)
            mean, std = np.mean(rewards), np.std(rewards)
            if mean > best_score:
                ql.save("agent.npy")
                best_score = mean
            print(f"Step: {i + 1}, Reward mean: {mean}, Reward std: {std}")


if __name__ == "__main__":
    main()
