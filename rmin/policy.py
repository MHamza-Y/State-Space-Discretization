import numpy as np

from base_algorithm.base_policies import Policy


class RMinPolicy(Policy):

    def get_action(self, state):
        return np.argmax(self.q_table[state])
