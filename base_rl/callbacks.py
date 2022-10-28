from abc import abstractmethod

import numpy as np


class Callback:
    def __call__(self, algo, trainer):
        self.execute_callback(algo=algo, trainer=trainer)

    @abstractmethod
    def execute_callback(self, algo, trainer):
        pass


class OnTrainingStartCallback(Callback):
    pass


class OnTrainingEndCallback(Callback):
    pass


class OnEpisodeStartCallback(Callback):
    pass


class OnEpisodeEndCallback(Callback):
    pass


class SaveBestPolicy(OnEpisodeEndCallback):

    def __init__(self, save_path):
        self.save_path = save_path

    def execute_callback(self, algo, trainer):
        last_rewards = algo.get_last_rewards()
        new_best_reward_received = np.max(last_rewards) == last_rewards[-1]

        if new_best_reward_received:
            algo.get_policy().save(self.save_path)


class SavePolicyOnTrainingEnd(OnTrainingEndCallback):

    def __init__(self, save_path):
        self.save_path = save_path

    def execute_callback(self, algo, trainer):
        algo.get_policy().save(self.save_path)
