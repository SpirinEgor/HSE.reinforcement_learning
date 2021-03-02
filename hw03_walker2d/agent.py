import pickle
import sys
from os.path import dirname

import torch

sys.path.append(dirname(__file__))

from actor_critic import Actor


class Agent:
    def __init__(self, path: str = "agent.pkl", seed: int = 7):
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        with open(__file__[:-8] + path, "rb") as in_file:
            state_dict = pickle.load(in_file)
        self._actor = Actor(**state_dict["init_params"])
        self._actor.load_state_dict(state_dict["state_dict"])
        self._actor.to(self._device)

        torch.manual_seed(seed)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32, device=self._device)
            action, _, _ = self._actor(state)
        return action[0].cpu().numpy()

    def reset(self):
        pass
