import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from base_rl.algorithm import RLAlgorithm
from base_rl.base_policies import DiscretePolicy
from base_rl.scheduler import HyperParamScheduler


class QLPolicy(DiscretePolicy):
    def state_exist(self, state):
        return state in self.q_table

    def __init__(self, action_space_n, q_table=None, initial_q_value=0):
        if q_table:
            self.q_table = q_table
        else:
            self.q_table = {}

        self.initial_q_value = initial_q_value
        self.action_space_n = action_space_n

    def get_action(self, state):
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, new_value):
        self.q_table[state][action] = new_value

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def get_max_q_value(self, state):
        return np.max(self.q_table[state])

    def add_new_state(self, state):
        if not self.state_exist(state):
            self.q_table[state] = np.ones(self.action_space_n) * self.initial_q_value


class QLearningAlgo(RLAlgorithm):

    def __init__(self, epochs, alpha: HyperParamScheduler, gamma, epsilon: HyperParamScheduler, env_creator, env_kwargs,
                 reward_offset=0,
                 show_reward_type='mean', policy=None, comment='', initial_q_value=0):

        self.epochs = epochs
        self.comment = comment
        self.current_epoch = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env_creator = env_creator
        self.env_kwargs = env_kwargs
        self.reward_offset = reward_offset
        self.show_reward_type = show_reward_type
        self.policy = policy
        self.mean_train_reward_per_epoch = []
        self.eval_rewards_per_epoch = []
        self.eval_new_state_found = []
        self.eval_trajectories = []
        self.initial_q_value = initial_q_value
        self.env = None
        self.writer = None

    def setup(self):
        self.writer = SummaryWriter(comment=self.comment)
        self.env = self.env_creator(**self.env_kwargs)
        self.mean_train_reward_per_epoch = []
        self.current_epoch = 0
        if not self.policy:
            self.policy = QLPolicy(self.env.action_space.n, initial_q_value=self.initial_q_value)

    def q_table_update_step(self, state, action, reward, next_state, alpha):
        old_value = self.policy.get_q_value(state=state, action=action)
        next_max = self.policy.get_max_q_value(next_state)

        new_value = (1 - alpha) * old_value + alpha * (
                reward + self.reward_offset + self.gamma * next_max)
        self.policy.update_q_value(state=state, action=action, new_value=new_value)

    def train_episode(self):
        state = self.env.reset()
        self.policy.add_new_state(state)
        done = False
        total_reward = 0
        total_steps = 0
        current_alpha = self.alpha.get()
        self.alpha.step()
        current_eps = self.epsilon.get()
        self.epsilon.step()
        while not done:

            if random.uniform(0, 1) < current_eps:
                action = self.env.action_space.sample()  # Explore action space
            else:
                action = self.policy.get_action(state)  # Exploit learned values

            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            self.policy.add_new_state(next_state)
            self.q_table_update_step(state=state, action=action, reward=reward, next_state=next_state,
                                     alpha=current_alpha)

            state = next_state

            total_steps += 1
        if self.show_reward_type == 'mean':
            self.mean_train_reward_per_epoch.append(total_reward / total_steps)
        else:
            self.mean_train_reward_per_epoch.append(total_reward)

        self.writer.add_scalar('Q-Learning/Mean Reward', self.mean_train_reward_per_epoch[-1], self.current_epoch)
        self.writer.add_scalar('Q-Learning/Epsilon', current_eps, self.current_epoch)
        self.writer.add_scalar('Q-Learning/Alpha', current_alpha, self.current_epoch)
        self.writer.add_scalar('Q-Learning/Total States', len(self.policy.q_table.keys()), self.current_epoch)
        self.writer.flush()
        self.current_epoch += 1

    def get_last_rewards(self):
        return self.mean_train_reward_per_epoch

    def get_policy(self):
        return self.policy

    def keep_training(self):
        return self.current_epoch < self.epochs
