import os

import dill
import numpy as np
import torch


class MDPModel:

    def __init__(self, states, next_states, actions, rewards, dones, device='cpu', reward_nan=-1000, sa_reward=False):
        self.device = device
        state_tr = np.array([states, next_states])

        self.index_to_state, state_tr_inverse_search = np.unique(state_tr, return_inverse=True)
        state_tr_inverse_search = np.array(np.split(state_tr_inverse_search, 2))

        state_inverse_search = state_tr_inverse_search[0]
        next_state_inverse_search = state_tr_inverse_search[1]

        self.index_to_actions, action_inverse_search = np.unique(actions, return_inverse=True)

        self.total_samples = state_inverse_search.shape[0]
        print('Computing Reward Function')
        if sa_reward:
            self.reward_function = self.compute_reward_table_sa(rewards=rewards,
                                                                current_state_inverse_search=state_inverse_search,
                                                                action_inverse_search=action_inverse_search)
        else:
            self.reward_function = self.compute_reward_table_s(rewards=rewards,
                                                               current_state_inverse_search=state_inverse_search)
        print('Computing Transition Model')
        self.N_D_sa = self.compute_state_action_pair_count()
        self.transition_model = self.compute_state_transition_model(state_inverse_search,
                                                                    action_inverse_search,
                                                                    next_state_inverse_search)
        self.reward_function = torch.nan_to_num(self.reward_function, nan=rewards.min()).numpy()
        self.transition_model = torch.nan_to_num(self.transition_model).numpy()

        self.state_to_index = {}
        for i, s in enumerate(self.index_to_state):
            self.state_to_index[s] = i

    def get_total_states_count(self):
        return self.index_to_state.shape[0]

    def get_total_actions_count(self):
        return self.index_to_actions.shape[0]

    def compute_reward_table_s(self, rewards, current_state_inverse_search):
        total_states = self.get_total_states_count()
        total_rewards = torch.zeros([total_states], device=self.device, dtype=torch.float32)
        state_encountered = torch.zeros([total_states], device=self.device, dtype=torch.int32)

        for i in range(self.total_samples):
            state_index = current_state_inverse_search[i]

            total_rewards[state_index] += rewards[i]
            state_encountered[state_index] += 1

        return total_rewards / state_encountered

    def compute_reward_table_sa(self, rewards, current_state_inverse_search, action_inverse_search):
        total_states = self.get_total_states_count()
        total_actions = self.get_total_actions_count()
        total_rewards = torch.zeros([total_states, total_actions], device=self.device)
        state_action_encountered = torch.zeros([total_states, total_actions], device=self.device, dtype=torch.int32)

        for i in range(self.total_samples):
            state_index = current_state_inverse_search[i]
            action_index = action_inverse_search[i]
            total_rewards[state_index, action_index] += rewards[i]
            state_action_encountered[state_index, action_index] += 1

        return total_rewards / state_action_encountered

    def compute_state_transition_model(self, state_inverse_search,
                                       action_inverse_search, next_state_inverse_search):
        total_states = self.get_total_states_count()
        total_actions = self.get_total_actions_count()

        total_state_action_next_state_transitions = torch.zeros(
            [total_states, total_actions, total_states], device=self.device, dtype=torch.int32)
        total_state_action_pair_encountered = torch.zeros([total_states, total_actions], device=self.device,
                                                          dtype=torch.int32)

        for i in range(self.total_samples):
            state_index = state_inverse_search[i]
            action_index = action_inverse_search[i]
            next_state_index = next_state_inverse_search[i]

            total_state_action_next_state_transitions[state_index, action_index, next_state_index] += 1
            total_state_action_pair_encountered[state_index, action_index] += 1

        return total_state_action_next_state_transitions / total_state_action_pair_encountered.view(
            total_states, total_actions, 1)

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, save_path):
        with open(save_path, 'rb') as f:
            return dill.load(f)

    def compute_state_action_pair_count(self):
        pass
