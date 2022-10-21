import os
import time
from typing import List

import gym
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from torch.multiprocessing import Process, set_start_method, Value

from base_algorithm.base_policies import RandomPolicy
from offline_dataset.convert_datset import merge_rllib_out, save_numpy

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def get_evenly_divided_values(value_to_be_distributed, times):
    return [value_to_be_distributed // times + int(x < value_to_be_distributed % times) for x in range(times)]


class GymEnvSamplerProcess(Process):
    def __init__(self, env_creator, env_kwargs, path, ep_count, episodes: int = 100, compress_columns=None,
                 policy=None, reward_threshold=None, buffer_transform=None, buffer_transform_kwargs=None):
        super().__init__()
        if compress_columns is None:
            self.compress_columns = []
        self.env_creator = env_creator
        self.env_kwargs = env_kwargs
        self.episodes = episodes
        self.path = path
        self.batch_builder = SampleBatchBuilder()
        self.writer = JsonWriter(
            self.path,
            compress_columns=self.compress_columns
        )
        self.ep_count = ep_count
        self.policy = policy
        self.reward_threshold = reward_threshold
        self.buffer_transform = buffer_transform
        self.buffer_transform_kwargs = buffer_transform_kwargs

    def run(self) -> None:
        self.env = self.env_creator(**self.env_kwargs)
        if self.policy is None:
            self.policy = RandomPolicy(self.env.action_space)

        eps_id = 0
        if callable(self.buffer_transform):
            buffer_transform = self.buffer_transform(**self.buffer_transform_kwargs)
        else:
            buffer_transform = self.buffer_transform
        while eps_id < self.episodes:
            obs = self.env.reset()
            done = False
            t = 0
            eps_total_reward = 0
            while not done:
                action = self.policy.get_action(obs)
                new_obs, rew, done, info = self.env.step(action)

                self.batch_builder.add_values(
                    obs=obs,
                    actions=action,
                    rewards=rew,
                    dones=done,
                    new_obs=new_obs,
                )
                obs = new_obs
                eps_total_reward += rew
                t += 1
            eps_avg_reward = eps_total_reward / t

            if self.reward_threshold and eps_avg_reward < self.reward_threshold:
                self.batch_builder.build_and_reset()
                continue
            else:
                if buffer_transform:
                    self.batch_builder.buffers = buffer_transform(self.batch_builder.buffers)
                self.writer.write(self.batch_builder.build_and_reset())
                self.ep_count.value = self.ep_count.value + 1
                eps_id += 1


def print_progress(processes: List[GymEnvSamplerProcess], ep_count):
    while True:
        time.sleep(1)
        any_process_alive = False
        for p in processes:
            if p.is_alive():
                any_process_alive = True
        print(f'Episodes Sampled: {ep_count.value}')
        if not any_process_alive:
            print(f'Sampling Finished')
            break


class GymParallelSampler:

    def __init__(self, env_creator, path, episodes: int, workers: int, env_kwargs, compress_columns=None, policy=None,
                 reward_threshold=None, buffer_transform=None, buffer_transform_kwargs=None):
        self.path = path
        self.episodes = episodes
        self.workers = workers
        self.divided_episodes = get_evenly_divided_values(episodes, workers)
        self.paths = [os.path.join(path, f'worker_{i}') for i in range(workers)]
        self.ep_count = Value('i', 0)
        self.sampler_processes = [
            GymEnvSamplerProcess(env_creator=env_creator, env_kwargs=env_kwargs, path=self.paths[i],
                                 episodes=self.divided_episodes[i],
                                 compress_columns=compress_columns, ep_count=self.ep_count, policy=policy,
                                 reward_threshold=reward_threshold, buffer_transform=buffer_transform,
                                 buffer_transform_kwargs=buffer_transform_kwargs) for i in
            range(workers)]
        self.print_process = Process(target=print_progress, args=(self.sampler_processes,))

    def sample(self):
        for p in self.sampler_processes:
            p.start()
        # self.print_process.start()
        print_progress(self.sampler_processes, self.ep_count)
        # for p in self.sampler_processes:
        #     p.join()
        # self.print_process.join()

    def create_merged_dataset(self, save_path=None):
        data_path = os.path.join(self.path, '*', '*.json')
        merged_data = merge_rllib_out(data_path)
        if save_path:
            print(save_path)
            save_numpy(save_path, merged_data)

        return merged_data


class GymEnvSampler:

    def __init__(self, env: gym.Env, path, episodes: int = 100, compress_columns=None):
        if compress_columns is None:
            self.compress_columns = []
        self.env = env
        self.episodes = episodes
        self.path = path

    def create_dataset(self):
        batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
        writer = JsonWriter(
            self.path,
            compress_columns=self.compress_columns
        )

        for eps_id in range(self.episodes):
            print(f"Episode: {eps_id}")
            obs = self.env.reset()

            done = False
            t = 0
            while not done:
                action = self.env.action_space.sample()
                new_obs, rew, done, info = self.env.step(action)

                batch_builder.add_values(
                    obs=obs,
                    actions=action,
                    rewards=rew,
                    dones=done,
                    new_obs=new_obs,
                )
                obs = new_obs
                t += 1
            writer.write(batch_builder.build_and_reset())
