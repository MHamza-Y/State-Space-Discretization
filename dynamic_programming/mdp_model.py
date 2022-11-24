import os

import dill
import numpy as np
import torch


class MDPModel:

    def __init__(self, states, next_states, actions, rewards, device='cpu', reward_function_type='state_action',
                 calc_reward_variance=False):
        self.device = device
        state_tr = np.array([states, next_states])

        self.index_to_state, state_tr_inverse_search = np.unique(state_tr, return_inverse=True)
        state_tr_inverse_search = np.array(np.split(state_tr_inverse_search, 2))

        state_inverse_search = state_tr_inverse_search[0]
        next_state_inverse_search = state_tr_inverse_search[1]

        self.index_to_actions, action_inverse_search = np.unique(actions, return_inverse=True)

        self.total_samples = state_inverse_search.shape[0]
        print('Computing Transition Model')
        self.transition_model, self.count_state_action, self.count_state_action_state = self.compute_state_transition_model(
            state_inverse_search,
            action_inverse_search,
            next_state_inverse_search)

        print('Computing Reward Function')
        if reward_function_type == 'state_action':
            self.reward_function = self.compute_reward_table_sa(rewards=rewards,
                                                                current_state_inverse_search=state_inverse_search,
                                                                action_inverse_search=action_inverse_search)
        elif reward_function_type == 'state':
            self.reward_function = self.compute_reward_table_s(rewards=rewards,
                                                               current_state_inverse_search=state_inverse_search)
        elif reward_function_type == 'state_action_state':
            self.reward_function = self.compute_reward_table_sas(rewards=rewards,
                                                                 current_state_inverse_search=state_inverse_search,
                                                                 action_inverse_search=action_inverse_search,
                                                                 next_state_inverse_search=next_state_inverse_search)
            if calc_reward_variance:
                self.reward_variance = self.compute_sas_reward_variance(rewards=rewards,
                                                                        current_state_inverse_search=state_inverse_search,
                                                                        action_inverse_search=action_inverse_search,
                                                                        next_state_inverse_search=next_state_inverse_search,
                                                                        count_state_action_state=self.count_state_action_state,
                                                                        mean_reward=self.reward_function)
        else:
            raise ValueError(
                'reward_function_type must be one of these options: state, state_action, state_action_state')

        self.reward_function = torch.nan_to_num(self.reward_function, nan=rewards.min()).numpy()
        self.transition_model = torch.nan_to_num(self.transition_model).numpy()
        self.count_state_action = torch.nan_to_num(self.count_state_action).numpy()
        self.count_state_action_state = torch.nan_to_num(self.count_state_action_state).numpy()
        # self.reward_variance = torch.nan_to_num(self.reward_variance).numpy()

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

    def compute_reward_table_sas(self, rewards, current_state_inverse_search, action_inverse_search,
                                 next_state_inverse_search):
        total_states = self.get_total_states_count()
        total_actions = self.get_total_actions_count()
        total_rewards = torch.zeros([total_states, total_actions, total_states], device=self.device)
        state_action_state_encountered = torch.zeros([total_states, total_actions, total_states], device=self.device,
                                                     dtype=torch.int32)

        for i in range(self.total_samples):
            state_index = current_state_inverse_search[i]
            action_index = action_inverse_search[i]
            next_state_index = next_state_inverse_search[i]
            total_rewards[state_index, action_index, next_state_index] += rewards[i]
            state_action_state_encountered[state_index, action_index, next_state_index] += 1

        return total_rewards / state_action_state_encountered

    def compute_sas_reward_variance(self, rewards, current_state_inverse_search, action_inverse_search,
                                    next_state_inverse_search, count_state_action_state, mean_reward):
        total_states = self.get_total_states_count()
        total_actions = self.get_total_actions_count()
        sum_of_mean_diff_square = torch.zeros([total_states, total_actions, total_states], device=self.device)

        for i in range(self.total_samples):
            state_index = current_state_inverse_search[i]
            action_index = action_inverse_search[i]
            next_state_index = next_state_inverse_search[i]
            sum_of_mean_diff_square[state_index, action_index, next_state_index] += (rewards[i] - mean_reward[
                state_index, action_index, next_state_index]) ** 2

        return sum_of_mean_diff_square / count_state_action_state

    def compute_state_transition_model(self, state_inverse_search,
                                       action_inverse_search, next_state_inverse_search):
        total_states = self.get_total_states_count()
        total_actions = self.get_total_actions_count()

        count_state_action_state = torch.zeros(
            [total_states, total_actions, total_states], device=self.device, dtype=torch.int32)
        count_state_action = torch.zeros([total_states, total_actions], device=self.device,
                                         dtype=torch.int32)

        for i in range(self.total_samples):
            state_index = state_inverse_search[i]
            action_index = action_inverse_search[i]
            next_state_index = next_state_inverse_search[i]

            count_state_action_state[state_index, action_index, next_state_index] += 1
            count_state_action[state_index, action_index] += 1

        return count_state_action_state / count_state_action.view(
            total_states, total_actions, 1), count_state_action, count_state_action_state

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, save_path):
        with open(save_path, 'rb') as f:
            return dill.load(f)


def create_mdp_models(load_path, mdp_save_path, reward_function_type, device, calc_reward_variance=False,
                      reward_offset=0):
    samples = np.load(load_path, allow_pickle=True)[()]
    print(mdp_save_path)
    print(samples['rewards'].mean())
    print(samples['rewards'].size)
    print(np.unique(samples['obs']).size)
    samples['rewards'] += reward_offset
    mdp_model = MDPModel(states=samples['obs'], next_states=samples['new_obs'], actions=samples['actions'],
                         rewards=samples['rewards'], device=device,
                         reward_function_type=reward_function_type, calc_reward_variance=calc_reward_variance)
    mdp_model.save(mdp_save_path)
