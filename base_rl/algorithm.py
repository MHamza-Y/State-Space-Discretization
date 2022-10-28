from abc import abstractmethod

from base_rl.base_policies import Policy


class RLAlgorithm:

    @abstractmethod
    def get_policy(self) -> Policy:
        pass

    @abstractmethod
    def keep_training(self):
        pass

    @abstractmethod
    def get_last_rewards(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train_episode(self):
        pass
