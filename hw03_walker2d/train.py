import random

import numpy as np
import torch
from gym import make

from hw03_walker2d.config import Config
from hw03_walker2d.ppo import PPO


class Trainer:
    def __init__(self, config: Config):
        self._config = config

    def seed_everything(self, env):
        env.seed(self._config.seed)
        random.seed(self._config.seed)
        np.random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)

    def compute_lambda_returns_and_gae(self, trajectory):
        lambda_returns = []
        gae = []
        last_lr = 0.
        last_v = 0.
        for _, _, r, _, v in reversed(trajectory):
            ret = r + self._config.gamma * (
                    last_v * (1 - self._config.lambda_) + last_lr * self._config.lambda_
            )
            last_lr = ret
            last_v = v
            lambda_returns.append(last_lr)
            gae.append(last_lr - v)

        # Each transition contains state, action, old action probability, value estimation and advantage estimation
        return [
            (s, a, p, v, adv)
            for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))
        ]

    def evaluate_policy(self, env, agent: PPO):
        self.seed_everything(env)
        returns = []
        for _ in range(self._config.eval_episodes):
            done = False
            state = env.reset()
            total_reward = 0.

            while not done:
                state, reward, done, _ = env.step(agent.act(state)[0])
                total_reward += reward
            returns.append(total_reward)
        return returns

    def sample_episode(self, env, agent: PPO):
        s = env.reset()
        d = False
        trajectory = []
        while not d:
            a, pa, p = agent.act(s)
            v = agent.get_value(s)
            ns, r, d, _ = env.step(a)
            trajectory.append((s, pa, r, p, v))
            s = ns
        return self.compute_lambda_returns_and_gae(trajectory)

    def train(self):
        env = make(self._config.env_name)
        self.seed_everything(env)
        ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], config=self._config)
        state = env.reset()
        episodes_sampled = 0
        steps_sampled = 0

        best_score = None
        for i in range(self._config.iterations):
            trajectories = []
            steps_cnt = 0

            while len(trajectories) < self._config.min_episodes_per_update or \
                    steps_cnt < self._config.min_transitions_per_update:
                trajectory = self.sample_episode(env, ppo)
                steps_cnt += len(trajectory)
                trajectories.append(trajectory)
            episodes_sampled += len(trajectories)
            steps_sampled += steps_cnt

            ppo.update(trajectories)

            if (i + 1) % (self._config.iterations // 100) == 0:
                rewards = self.evaluate_policy(env, ppo)
                mean, std = np.mean(rewards), np.std(rewards)
                print(f"Step: {i + 1},"
                      f"Reward mean: {mean},"
                      f"Reward std: {std},"
                      f"Episodes: {episodes_sampled},"
                      f"Steps: {steps_sampled}")
                if best_score is None or mean - std > best_score:
                    ppo.save()
                    best_score = mean - std


if __name__ == "__main__":
    _trainer = Trainer(Config())
    _trainer.train()
