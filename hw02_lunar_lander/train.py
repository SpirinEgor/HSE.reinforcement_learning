import random
from copy import deepcopy

import numpy as np
import torch
from gym import make
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

SEED = 7

GAMMA = 0.99
INITIAL_STEPS = 4096
TRANSITIONS = 500_000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 256
LEARNING_RATE = 5e-4

START_EPS = 0.3
HIDDEN_DIM = 1024
N_LAYERS = 5
BUFFER_SIZE = INITIAL_STEPS
GRADIENT_CLIP = 5


class DQN:
    def __init__(self, state_dim, action_dim, device):
        self.steps = 0  # Do not change
        self._device = device

        self._position = 0
        self._state_buffer = torch.empty((BUFFER_SIZE, state_dim), dtype=torch.float, device=self._device)
        self._next_state_buffer = torch.empty((BUFFER_SIZE, state_dim), dtype=torch.float, device=self._device)
        self._action_buffer = torch.empty((BUFFER_SIZE, 1), dtype=torch.long, device=self._device)
        self._reward_buffer = torch.empty((BUFFER_SIZE, 1), dtype=torch.float, device=self._device)
        self._done_buffer = torch.empty((BUFFER_SIZE, 1), dtype=torch.bool, device=self._device)

        modules = [nn.Linear(state_dim, HIDDEN_DIM), nn.LeakyReLU()]
        for _ in range(N_LAYERS):
            modules += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LeakyReLU()]
        modules += [nn.Linear(HIDDEN_DIM, action_dim)]
        self.model = nn.Sequential(*modules).to(self._device)  # Torch model
        self._target_model = deepcopy(self.model)
        self._optimizer = Adam(self.model.parameters(), LEARNING_RATE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        state, action, next_state, reward, done = transition
        self._state_buffer[self._position] = torch.tensor(state, device=self._device)
        self._next_state_buffer[self._position] = torch.tensor(next_state, device=self._device)
        self._action_buffer[self._position] = action
        self._reward_buffer[self._position] = reward
        self._done_buffer[self._position] = done
        self._position = (self._position + 1) % BUFFER_SIZE

    def sample_batch(self):
        # Sample batch from a replay buffer.
        batch_idx = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)
        return self._state_buffer[batch_idx], self._action_buffer[batch_idx], self._next_state_buffer[batch_idx], \
            self._reward_buffer[batch_idx], self._done_buffer[batch_idx]

    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, action, next_state, reward, done = batch
        row_range = torch.arange(state.shape[0]).reshape(-1, 1)
        with torch.no_grad():
            # [batch size; action size]
            target_q = self._target_model(next_state)
            # [batch size; 1]
            target_q = torch.max(target_q, dim=1, keepdim=True)[0]
            target_q[done] = 0
            target_q = reward + GAMMA * target_q
        # [batch size; action size]
        q = self.model(state)
        # [batch size; 1]
        q = q[row_range, action]

        loss = F.mse_loss(q, target_q)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), GRADIENT_CLIP)
        self._optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self._target_model.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        with torch.no_grad():
            logits = self.model(torch.tensor(state, device=self._device))
            return logits.argmax(-1).numpy()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


def main(device):
    env = make("LunarLander-v2")
    env.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, device=device)
    eps = np.linspace(START_EPS, 0, TRANSITIONS)
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    best_score = None
    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        if random.random() < eps[i]:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            mean_reward = np.mean(rewards)
            print(f"Step: {i + 1}, Reward mean: {mean_reward}, Reward std: {np.std(rewards)}")
            if best_score is None or mean_reward > best_score:
                dqn.save()
                best_score = mean_reward


if __name__ == "__main__":
    main(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
