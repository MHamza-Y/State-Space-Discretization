import os

import numpy as np

from offline.convert_datset import merge_rllib_out_filtered, save_numpy


class RewardFilter:
    def __init__(self, reward_threshold):
        self.reward_threshold = reward_threshold

    def __call__(self, transitions):
        rewards = transitions['rewards']
        mean_reward = np.mean(rewards)
        return mean_reward > self.reward_threshold


save_path = os.path.join("tmp", "ibq-out/ibq_samples.npy")
reward_filter = RewardFilter(-250)
data_path = os.path.join("tmp", "ibq-out", '*', '*.json')
merged_data = merge_rllib_out_filtered(data_path, reward_filter)
if save_path:
    print(save_path)
    save_numpy(save_path, merged_data)
