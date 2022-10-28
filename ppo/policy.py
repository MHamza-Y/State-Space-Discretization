import numpy as np
from ray.rllib.agents.ppo import PPOTrainer

from base_rl.base_policies import Policy


class LSTMPPOPolicy(Policy):

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.config["num_workers"] = 1
        self.checkpoint_path = checkpoint_path
        self.cell_size = config["model"]["lstm_cell_size"]

    def load_agent(self):
        self.agent = PPOTrainer(config=self.config, env=None)
        self.agent.restore(self.checkpoint_path)
        self.state = [np.zeros(self.cell_size, np.float32),
                      np.zeros(self.cell_size, np.float32)]

    def get_action(self, obs):
        try:
            self.agent
        except AttributeError:
            self.load_agent()
        action, state_out, _ = self.agent.compute_action(obs[0:6], self.state, unsquash_action=True, clip_action=True,
                                                         explore=False)
        return action


class PPOPolicy(Policy):

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.config["num_workers"] = 1
        self.checkpoint_path = checkpoint_path

    def load_agent(self):
        self.agent = PPOTrainer(config=self.config, env=None)
        self.agent.restore(self.checkpoint_path)

    def get_action(self, obs):
        try:
            self.agent
        except AttributeError:
            self.load_agent()
        action = self.agent.compute_action(obs[0:6], full_fetch=False, explore=False)
        return action
