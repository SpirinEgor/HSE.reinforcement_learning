import torch
from torch.optim import Adam

from hw03_walker2d.actor_critic import Actor, Critic
from hw03_walker2d.config import Config


class PPO:

    _eps = 1e-8

    def __init__(self, state_dim: int, action_dim: int, config: Config):
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._config = config

        self.actor = Actor(state_dim, action_dim, config).to(self._device)
        self.critic = Critic(state_dim, config).to(self._device)
        self.actor_optimizer = Adam(self.actor.parameters(), config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), config.critic_lr)

    def update(self, trajectories):
        # Turn a list of trajectories into list of transitions
        transitions = [t for trajectory in trajectories for t in trajectory]
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = torch.tensor(state, dtype=torch.float32, device=self._device)   # [n transitions; state dim]
        action = torch.tensor(state, dtype=torch.float32, device=self._device)  # [n transitions; action dim]
        # Probability of the action in state s.t. old policy
        old_prob = torch.tensor(old_prob, dtype=torch.float32, device=self._device)     # [n transitions; 1]
        # Estimated by lambda-returns
        target_value = torch.tensor(target_value, dtype=torch.float32, device=self._device)     # [n transitions; 1]
        # Estimated by generalized advantage estimation
        advantage = torch.tensor(advantage, dtype=torch.float32, device=self._device)   # [n transitions; 1]
        advantage = (advantage - advantage.mean()) / (advantage.std() + self._eps)

        for _ in range(self._config.batches_per_update):
            batch_idx = torch.randint(0, len(transitions), self._config.batch_size)     # Choose random batch
            batch_state = state[batch_idx]
            batch_action = action[batch_idx]
            batch_old_prob = old_prob[batch_idx]
            batch_target_value = target_value[batch_idx]
            batch_advantage = advantage[batch_idx]

            # TODO: Update actor here
            # TODO: Update critic here

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self._device)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self._device)
            action, pure_action, distribution = self.actor.act(state)
            prob = torch.exp(distribution.log_prob(pure_action).sum(-1))
        return action[0].cpu().item(), pure_action[0].cpu().item(), prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")
