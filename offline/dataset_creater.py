import gym
import numpy as np

from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter


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
            prev_action = np.zeros_like(self.env.action_space.sample())
            prev_reward = 0
            done = False
            t = 0
            while not done:
                action = self.env.action_space.sample()
                new_obs, rew, done, info = self.env.step(action)

                batch_builder.add_values(
                    t=t,
                    eps_id=eps_id,
                    agent_index=0,
                    obs=obs,
                    actions=action,
                    action_prob=1.0,  # put the true action probability here
                    action_logp=0.0,
                    rewards=rew,
                    prev_actions=prev_action,
                    prev_rewards=prev_reward,
                    dones=done,
                    infos=info,
                    new_obs=new_obs,
                )
                obs = new_obs
                prev_action = action
                prev_reward = rew
                t += 1
            writer.write(batch_builder.build_and_reset())
