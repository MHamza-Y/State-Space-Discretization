from ray.rllib.agents.ppo import PPOTrainer
import ray.tune as tune
import numpy as np

import industrial_benchmarks_train as ibt
from IBGym_modified import IBGymModded

import matplotlib.pyplot as plt

ibt.config["num_workers"] = 1
agent = PPOTrainer(config=ibt.config, env="IBGym-v1")


def test():
    episode_reward = 0
    done = False
    obs = env.reset()
    cell_size = ibt.config["model"]["lstm_cell_size"]
    state = [np.zeros(cell_size, np.float32),
             np.zeros(cell_size, np.float32)]
    while not done:
        action, state, logits = agent.compute_action(obs, state)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        print(info)
        env.render()
        episode_reward += reward

    return episode_reward


analysis = tune.ExperimentAnalysis('tmp/ray_exp_logs')

last_checkpoint = analysis.get_last_checkpoint(metric="episode_reward_mean", mode="max")

agent.restore(last_checkpoint)
env = ibt.env_creator('')

while True:
    test()
