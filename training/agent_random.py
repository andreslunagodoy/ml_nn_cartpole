import random
from training.agent_base import BaseAgent
from config import ACTION_DIM


class RandomAgent(BaseAgent):
    def __init__(self, action_dim=ACTION_DIM):
        self.action_dim = action_dim

    def select_action(self, state):
        return random.randint(0, self.action_dim - 1)

    def store_transition(self, s, a, r, s_next, done):
        pass

    def update(self):
        pass

    def update_target(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
