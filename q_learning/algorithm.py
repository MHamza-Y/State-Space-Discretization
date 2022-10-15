import random
import time
from typing import List

import numpy as np
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from torch.multiprocessing import Process, set_start_method, Manager

from base_algorithm.base_policies import Policy

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class QLPolicy(Policy):
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

    def state_in_q_table(self, state):
        return state in self.q_table

    def add_new_state(self, state):
        if not self.state_in_q_table(state):
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

    def evaluate(self, epochs, env_creator, env_kwargs, render=False, show_reward_type='mean'):
        self.eval_trajectories = []
        env = env_creator(**env_kwargs)
        self.eval_rewards_per_epoch = []
        for i in range(epochs):
            episode = {'obs': [], 'next_obs': [], 'rewards': [], 'actions': []}
            state = env.reset()

            new_state_found = False
            done = False
            total_reward = 0
            total_steps = 0
            while not done:
                episode['obs'].append(state)
                if self.policy.state_in_q_table(state):
                    action = self.policy.get_action(state)
                else:
                    action = env.action_space.sample()
                    new_state_found = True

                episode['actions'].append(action)
                state, reward, done, info = env.step(action)
                episode['next_obs'].append(state)
                episode['rewards'].append(reward)

                if render:
                    env.render()

                epochs += 1

                total_reward += reward
                total_steps += 1

            if show_reward_type is 'mean':
                self.eval_rewards_per_epoch.append(total_reward / total_steps)
            else:
                self.eval_rewards_per_epoch.append(total_reward)
            self.eval_trajectories.append(episode)
            self.eval_new_state_found.append(new_state_found)
            clear_output(wait=False)
            print(f'Episode {i} Reward: {self.eval_rewards_per_epoch[-1]}')
            plt.plot(self.eval_rewards_per_epoch)
            plt.show()


class QLTrainingProcess(Process):

    def __init__(self, qlearning_algo, worker_training_kwargs):
        super().__init__()
        self.qlearning_algo = qlearning_algo
        self.worker_training_kwargs = worker_training_kwargs

    def run(self) -> None:
        self.qlearning_algo.train(**self.worker_training_kwargs)
        return self.qlearning_algo


class QLEvaluationProcess(Process):
    def __init__(self, qlearning_algo, workers_evaluation_kwargs):
        super().__init__()
        self.qlearning_algo = qlearning_algo
        self.workers_evaluation_kwargs = workers_evaluation_kwargs

    def run(self) -> None:
        self.qlearning_algo.evaluate(**self.workers_evaluation_kwargs)


def monitor_training_process(processes: List[QLTrainingProcess], q_learning_algos, tags):
    while True:
        time.sleep(2)
        any_process_alive = False
        # clear_output(wait=True)
        for i, p in enumerate(processes):
            if p.is_alive():
                any_process_alive = True
            mean_train_reward_per_epoch = q_learning_algos[i].mean_train_reward_per_epoch
            episode = len(mean_train_reward_per_epoch)
            print(tags[i])
            print(episode)
            if episode > 1:
                print(f'Episode: {episode} Reward: {mean_train_reward_per_epoch[-1]}')
                plt.plot(mean_train_reward_per_epoch)
                plt.show()
        if not any_process_alive:
            print('Training Finished')
            break


def monitor_eval_process(processes: List[QLEvaluationProcess], q_learning_algos, tags):
    while True:
        time.sleep(2)
        any_process_alive = False
        for i, p in enumerate(processes):

            if p.is_alive():
                any_process_alive = True
            eval_rewards_per_epoch = q_learning_algos[i].eval_rewards_per_epoch
            eval_new_state_found = q_learning_algos[i].eval_new_state_found
            episode = len(eval_rewards_per_epoch)
            print(tags[i])
            if episode > 1:
                print(f'Episode: {episode} Reward: {eval_rewards_per_epoch[-1]}')
                print(f'Unknown Action: {eval_new_state_found[-1]}')
                plt.plot(eval_rewards_per_epoch)
                plt.show()
                no_unknown_state_rewards = [rew for rew, state_found in
                                            zip(eval_rewards_per_epoch, eval_new_state_found) if not state_found]
                some_unknown_state_rewards = [rew for rew, state_found in
                                              zip(eval_rewards_per_epoch, eval_new_state_found) if state_found]

                print(f'No unknown States Found mean reward: {np.mean(no_unknown_state_rewards)}')
                print(f'Some unknown States Found mean reward: {np.mean(some_unknown_state_rewards)}')

        if not any_process_alive:
            print('Evaluation Finished')
            break


class QLParallelExperiment:

    def __init__(self, workers_training_kwargs, workers_evaluation_kwargs, tags):
        self.mgr = Manager()
        self.tags = tags
        self.qlearning_algos = self.mgr.list([QLearning() for i in range(len(workers_training_kwargs))])

        self.training_processes = [
            QLTrainingProcess(
                self.qlearning_algos[i],
                train_kwarg
            ) for i, train_kwarg in enumerate(workers_training_kwargs)]

        self.evaluation_processes = []
        self.workers_evaluation_kwargs = workers_evaluation_kwargs

    def start_training(self):
        for train_process in self.training_processes:
            train_process.start()

        monitor_training_process(self.training_processes, self.qlearning_algos, self.tags)

    def start_evaluation(self):
        for i, algo in enumerate(self.qlearning_algos):
            self.evaluation_processes.append(
                QLEvaluationProcess(self.qlearning_algos[i], self.workers_evaluation_kwargs)
            )

        for eval_process in self.evaluation_processes:
            eval_process.start()

        monitor_eval_process(self.evaluation_processes)
