import os

import dill
import gym


class Policy:
    def get_action(self, state):
        raise NotImplementedError

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, save_path):
        with open(save_path, 'rb') as f:
            return dill.load(f)


class DiscretePolicy(Policy):
    def state_exist(self, state):
        raise NotImplementedError


class RandomPolicy(Policy):

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def get_action(self, state):
        return self.action_space.sample()
