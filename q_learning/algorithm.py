import random

import numpy as np
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt

from base_algorithm.base_policies import DiscretePolicy


class QLPolicy(DiscretePolicy):
    def state_exist(self, state):
        return state in self.q_table

    def __init__(self, action_space_n, q_table=None, ):
        if q_table:
            self.q_table = q_table
        else:
            self.q_table = {}

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
            self.q_table[state] = np.zeros(self.action_space_n)


class QLearning:

    def __init__(self, policy=None):
        self.policy = policy
        self.mean_train_reward_per_epoch = []
        self.eval_rewards_per_epoch = []
        self.eval_new_state_found = []
        self.eval_trajectories = []

    def train(self, epochs, alpha, gamma, epsilon, env_creator, env_kwargs, reward_offset=0, graph=True,
              show_reward_type='mean'):
        epsilon = np.array(epsilon)
        epsilon_is_arr = epsilon.size > 1
        alpha = np.array(alpha)
        alpha_is_arr = alpha.size > 1
        env = env_creator(**env_kwargs)
        self.mean_train_reward_per_epoch = []
        if not self.policy:
            self.policy = QLPolicy(env.action_space.n)

        for i in range(epochs):

            state = env.reset()
            self.policy.add_new_state(state)
            done = False
            total_reward = 0
            total_steps = 0
            if alpha_is_arr:
                if i < alpha.size:
                    current_alpha = alpha[i]
                else:
                    current_alpha = alpha[-1]
            else:
                current_alpha = alpha

            if epsilon_is_arr:
                if i < epsilon.size:
                    current_eps = epsilon[i]
                else:
                    current_eps = epsilon[-1]
            else:
                current_eps = epsilon

            while not done:

                if random.uniform(0, 1) < current_eps:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = self.policy.get_action(state)  # Exploit learned values

                next_state, reward, done, info = env.step(action)
                total_reward += reward

                self.policy.add_new_state(next_state)
                old_value = self.policy.get_q_value(state=state, action=action)
                next_max = self.policy.get_max_q_value(next_state)

                new_value = (1 - current_alpha) * old_value + current_alpha * (
                        reward + reward_offset + gamma * next_max)
                self.policy.update_q_value(state=state, action=action, new_value=new_value)

                state = next_state

                total_steps += 1
            if show_reward_type is 'mean':
                self.mean_train_reward_per_epoch.append(total_reward / total_steps)
            else:
                self.mean_train_reward_per_epoch.append(total_reward)
            if graph:
                clear_output(wait=False)
                plt.plot(self.mean_train_reward_per_epoch)
                plt.show()
            print(f'Episode {i} Reward: {self.mean_train_reward_per_epoch[-1]}')
            print(f'Total States: {len(self.policy.q_table.keys())}')





