import gym

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
