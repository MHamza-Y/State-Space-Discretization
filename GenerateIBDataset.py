import os
import shutil

from IBGym_modified import IBGymModded
from offline.convert_datset import create_lstm_dataset
from offline.dataset_creater import GymEnvSampler


def env_creator(steps_per_episode=1000):
    return IBGymModded(70, reward_type='classic', action_type='continuous', reset_after_timesteps=steps_per_episode)


if __name__ == "__main__":
    episodes = 1000
    steps_per_episode = 1000
    input_time_steps = 50
    output_time_steps = 10

    writer_path = os.path.join("tmp", "ib-out")
    if os.path.exists(writer_path) and os.path.isdir(writer_path):
        shutil.rmtree(writer_path)
    env = env_creator(steps_per_episode)
    env_sampler = GymEnvSampler(env, writer_path, episodes=episodes)
    env_sampler.create_dataset()
    create_lstm_dataset("tmp/ib-out/*.json", "tmp/ib-out/ib-samples.json", input_steps=3, output_steps=2)
