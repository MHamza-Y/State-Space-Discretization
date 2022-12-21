import matplotlib.pyplot as plt
import numpy as np


def get_policy(q):
    return np.argmax(q, axis=1)


def get_policy_changes(old_q_table, new_q_table):
    pi_old = get_policy(old_q_table)
    pi_new = get_policy(new_q_table)
    return np.count_nonzero(pi_new != pi_old)


class RMinTrainer:

    def __init__(self, reward_function, transition_model, count_state_action, min_count):
        self.reward_function = reward_function
        self.transition_model = transition_model
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.count_state_action = count_state_action
        self.min_count = min_count
        self.r_min = np.min(reward_function)
        self._compute_mask()
        self.q_table = np.zeros(shape=(self.num_states, self.num_actions))

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def single_policy_evaluation(self, old_q_table, gamma):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                if self.mask[state, action]:
                    p = self.transition_model[state, action]
                    values = np.max(old_q_table, axis=1)
                    future_return = np.dot(p, values)
                    self.q_table[state, action] = self.reward_function[state, action] + gamma * future_return

    def train(self, epochs, plot=True, gamma=0.999):
        self.q_table[~self.mask] = self.r_min / (1 - gamma)
        policy_changes = []
        for epoch in range(epochs):
            old_q_table = np.array(self.q_table, copy=True)
            self.single_policy_evaluation(old_q_table=old_q_table, gamma=gamma)
            policy_changes.append(get_policy_changes(old_q_table=old_q_table, new_q_table=self.q_table))
            print(f'Old vs New policy difference: {np.linalg.norm(self.q_table - old_q_table)}')
            # if policy_changes[-1] == 0:
            #     print(f'Epoch: {epoch}')
            #     break

        if plot:
            plt.plot(policy_changes, marker='o',
                     markersize=4, alpha=0.7, color='#d62728',
                     label='# policy updates in \npolicy improvement\n' + r'$\gamma =$' + f'{gamma}')
            plt.xlabel('Epoch')
            plt.show()

    def _compute_mask(self):

        self.mask = self.count_state_action > self.min_count
