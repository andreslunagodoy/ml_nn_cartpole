from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_next, done):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def update_target(self):
        pass

    def decay_epsilon(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
