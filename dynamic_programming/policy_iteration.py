import time

import matplotlib.pyplot as plt
import numpy as np
import torch


class PolicyIteration:
    def __init__(self, reward_function, transition_model, gamma, init_policy=None, sa_reward=False):
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.reward_function = reward_function

        self.transition_model = transition_model
        self.gamma = gamma
        self.sa_reward = sa_reward

        self.values = np.zeros(self.num_states)
        if init_policy is None:
            self.policy = np.random.randint(0, self.num_actions, self.num_states)
        else:
            self.policy = init_policy

    def one_policy_evaluation(self):
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            a = self.policy[s]
            p = self.transition_model[s, a]
            if self.sa_reward:
                self.values[s] = self.reward_function[s][a] + self.gamma * np.sum(p * self.values)
            else:
                self.values[s] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def run_policy_evaluation(self, eval_epochs, tol=1e-3):
        epoch = 0
        delta = self.one_policy_evaluation()
        delta_history = [delta]
        while epoch < eval_epochs:
            epoch += 1
            delta = self.one_policy_evaluation()
            delta_history.append(delta)
            print(delta)
            if delta < tol:
                break
        return len(delta_history)

    def run_policy_improvement(self):
        update_policy_count = 0
        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = np.sum(p * self.values)
            self.policy[s] = np.argmax(v_list)
            if temp != self.policy[s]:
                update_policy_count += 1
        return update_policy_count

    def train(self, tol=1e-4, plot=True, total_epochs=500, eval_epochs=500):
        epoch = 0
        eval_count = self.run_policy_evaluation(tol=tol, eval_epochs=eval_epochs)
        eval_count_history = [eval_count]
        policy_change = self.run_policy_improvement()
        policy_change_history = [policy_change]
        while epoch < total_epochs:
            print(epoch)
            epoch += 1
            new_eval_count = self.run_policy_evaluation(tol=tol, eval_epochs=eval_epochs)
            new_policy_change = self.run_policy_improvement()
            eval_count_history.append(new_eval_count)
            policy_change_history.append(new_policy_change)
            if new_policy_change == 0:
                break

        print(f'# epoch: {len(policy_change_history)}')
        print(f'eval count = {eval_count_history}')
        print(f'policy change = {policy_change_history}')

        if plot is True:
            fig, axes = plt.subplots(2, 1, figsize=(3.5, 4), sharex='all', dpi=200)
            axes[0].plot(np.arange(len(eval_count_history)), eval_count_history, marker='o', markersize=4, alpha=0.7,
                         color='#2ca02c', label='# sweep in \npolicy evaluation\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[0].legend()

            axes[1].plot(np.arange(len(policy_change_history)), policy_change_history, marker='o',
                         markersize=4, alpha=0.7, color='#d62728',
                         label='# policy updates in \npolicy improvement\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            plt.tight_layout()
            plt.show()


class PolicyIterationTorch:
    def __init__(self, reward_function, transition_model, gamma, init_policy=None, device='cpu'):
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.reward_function = torch.nan_to_num(reward_function)

        self.transition_model = transition_model
        self.gamma = gamma
        self.device = device
        self.values = torch.zeros(self.num_states, device=self.device)
        if init_policy is None:
            self.policy = torch.from_numpy(np.random.randint(0, self.num_actions, self.num_states)).to(self.device)
        else:
            self.policy = init_policy

    def one_policy_evaluation(self):
        delta = torch.tensor(0)
        for s in range(self.num_states):
            temp = self.values[s]
            a = self.policy[s]
            p = self.transition_model[s, a]
            self.values[s] = self.reward_function[s] + self.gamma * torch.sum(p * self.values)
            delta = torch.max(delta, torch.abs(temp - self.values[s]))
            print(torch.abs(temp - self.values[s]))
            print('>')
            print(delta)
        return delta

    def run_policy_evaluation(self, tol=1e-3):
        epoch = 0
        delta = self.one_policy_evaluation()
        delta_history = [delta]
        while epoch < 100:
            delta = self.one_policy_evaluation()
            delta_history.append(delta)
            epoch += 1
            # if delta < tol:
            #     break
        return len(delta_history)

    def run_policy_improvement(self):
        update_policy_count = 0
        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = torch.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = torch.sum(p * self.values)
            self.policy[s] = torch.argmax(v_list)
            if temp != self.policy[s]:
                update_policy_count += 1
        return update_policy_count

    def train(self, tol=1e-3, plot=True, total_epoch=500):
        epoch = 0

        eval_count = self.run_policy_evaluation(tol=tol)

        eval_count_history = [eval_count]
        policy_change = self.run_policy_improvement()
        policy_change_history = [policy_change]
        while epoch < total_epoch:
            start = time.time()
            print(epoch)
            epoch += 1
            new_eval_count = self.run_policy_evaluation(tol)
            new_policy_change = self.run_policy_improvement()
            end = time.time()
            print(end - start)
            eval_count_history.append(new_eval_count)
            policy_change_history.append(new_policy_change)
            if new_policy_change == 0:
                break

        print(f'# epoch: {len(policy_change_history)}')
        print(f'eval count = {eval_count_history}')
        print(f'policy change = {policy_change_history}')

        if plot is True:
            fig, axes = plt.subplots(2, 1, figsize=(3.5, 4), sharex='all', dpi=200)
            axes[0].plot(np.arange(len(eval_count_history)), eval_count_history, marker='o', markersize=4, alpha=0.7,
                         color='#2ca02c', label='# sweep in \npolicy evaluation\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[0].legend()

            axes[1].plot(np.arange(len(policy_change_history)), policy_change_history, marker='o',
                         markersize=4, alpha=0.7, color='#d62728',
                         label='# policy updates in \npolicy improvement\n' + r'$\gamma =$' + f'{self.gamma}')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            plt.tight_layout()
            plt.show()
