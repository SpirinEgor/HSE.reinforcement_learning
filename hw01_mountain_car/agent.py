from os.path import join

import numpy as np

from .train import transform_state


class Agent:
    def __init__(self, path="agent.npy"):
        full_path = join(__file__[:-8], path)
        with open(full_path, "rb") as f_in:
            self.qlearning_estimate = np.load(f_in)
        
    def act(self, state):
        state = transform_state(state)
        return self.qlearning_estimate[state].argmax()

    def reset(self):
        pass
