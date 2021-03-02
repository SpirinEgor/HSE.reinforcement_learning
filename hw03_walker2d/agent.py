import sys
from os.path import dirname

import torch

sys.path.append(dirname(__file__))

from actor_critic import Actor


class Agent:
    def __init__(self, path: str = "agent.pkl", seed: int = 7):
        self._device = torch.device("cpu")
        state_dict = torch.load(__file__[:-8] + path, map_location=self._device)
        self._actor = Actor(**state_dict["init_params"])
        self._actor.load_state_dict(state_dict["state_dict"])

        torch.manual_seed(seed)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32, device=self._device)
            action, _, _ = self._actor(state)
        return action[0].cpu().numpy()

    def reset(self):
        pass
