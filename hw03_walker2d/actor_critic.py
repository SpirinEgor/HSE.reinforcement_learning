import torch
from torch import nn
from torch.distributions import Normal

from hw03_walker2d.config import Config


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        super().__init__()
        modules = [nn.Linear(state_dim, config.actor_hidden_dim), nn.ELU()]
        for _ in range(config.actor_hidden_layers):
            modules += [nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim), nn.ELU()]
        modules += [nn.Linear(config.actor_hidden_dim, action_dim)]
        self.model = nn.Sequential(*modules)
        self.sigma = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        # (use it to compute entropy loss)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distribution = Normal(mu, sigma)
        return torch.exp(distribution.log_prob(action).sum(-1)), distribution

    def forward(self, state):
        # Returns an action, not-transformed action and distribution
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distribution = Normal(mu, sigma)
        pure_action = distribution.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, distribution


class Critic(nn.Module):
    def __init__(self, state_dim: int, config: Config):
        super().__init__()

        modules = [nn.Linear(state_dim, config.critic_hidden_dim), nn.ELU()]
        for _ in range(config.critic_hidden_layers):
            modules += [nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim), nn.ELU()]
        modules += [nn.Linear(config.critic_hidden_dim, 1)]
        self.model = nn.Sequential(*modules)

    def forward(self, state):
        return self.model(state)
